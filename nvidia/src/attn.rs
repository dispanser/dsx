//! Prefill attention on tensor cores: launch helper for `sparse_attn_tc` / `sparse_attn_tc_fast` (dsx.cu).
//! `sparse_attn` stays for decode (T = 1) and as the reference. T = 4096 on the 5090: 22.2 -> 4.2 ms (fast: 2.6 ms).
//! SM70 (arch < 80, Volta) has no mma/ldmatrix/cp.async and the tc kernels are traps there: `launch` routes to the
//! CUDA-core `sparse_attn` instead — same contract and arguments, grid (T, H/8), no dynamic smem.
use crate::gpu::{Arg, Gpu, Kernel};
use anyhow::{Result, ensure};

/// rows per smem tile (TC_NT in dsx.cu)
const NT: usize = 32;
/// opt-in dynamic smem limit per block on sm_86 and sm_120
pub const MAX_SMEM: usize = 99 * 1024;

/// dynamic smem for `rows` = W + K candidate rows per token: 2 row tiles, partial scores, P (hi, lo), corr/l, scan, row list
pub fn smem_bytes(rows: usize) -> usize {
    2 * NT * 512 * 2 + 4 * 32 * (NT + 4) * 4 + 2 * 32 * (NT + 8) * 2 + 64 * 4 + 8 * 4 + rows * 4
}

pub struct SparseAttnTc {
    k: Kernel,
    kf: Kernel,
    /// packed-FP4 variants of k / kf, selected when the compressed rows are packed (DSX_KV_FP4)
    k_fp4: Kernel,
    kf_fp4: Kernel,
    /// arch < 80: the CUDA-core `sparse_attn` replaces the tensor-core kernels (same 12 arguments)
    serial: Option<Kernel>,
    /// use sparse_attn_tc_fast: q and p as single bf16 (2x fewer mma, ~bf16-level error on peaky softmax)
    pub fast: bool,
    /// the compressed rows are packed FP4 (16 code bytes + 2 E4M3 scales per 32 values)
    pub fp4: bool,
}

impl SparseAttnTc {
    /// looks the kernels up and raises their dynamic smem limit (once per Gpu)
    pub fn new(g: &Gpu) -> Result<Self> {
        let (k, kf) = (
            g.kernel("sparse_attn_tc")?,
            g.kernel("sparse_attn_tc_fast")?,
        );
        let (k_fp4, kf_fp4) = (
            g.kernel("sparse_attn_tc_fp4")?,
            g.kernel("sparse_attn_tc_fast_fp4")?,
        );
        // Volta: the mma kernels do not exist (dsx.cu traps on them), and its 96 KB dynamic-smem ceiling is below
        // MAX_SMEM, so no limit raise. launch() sends every call to `sparse_attn`.
        if g.arch < 80 {
            return Ok(SparseAttnTc {
                k,
                kf,
                k_fp4,
                kf_fp4,
                serial: Some(g.kernel("sparse_attn")?),
                fast: false,
                fp4: false,
            });
        }
        for x in [k, kf, k_fp4, kf_fp4] {
            g.set_smem(x, MAX_SMEM)?;
        }
        Ok(SparseAttnTc {
            k,
            kf,
            k_fp4,
            kf_fp4,
            serial: None,
            fast: false,
            fp4: false,
        })
    }

    /// o[T][H][512] = sparse MQA attention of q[T][H][512] (f32) over win[win_idx[t][..W]] ++ comp[comp_idx[t][..K]]
    /// (bf16 rows, -1 = skip; comp may be 0 when K = 0), sinks[H]; same arguments as `sparse_attn`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        g: &Gpu,
        q: u64,
        win: u64,
        win_idx: u64,
        w: usize,
        comp: u64,
        comp_idx: u64,
        k: usize,
        sinks: u64,
        o: u64,
        t: usize,
        nh: usize,
        hd: usize,
    ) -> Result<()> {
        ensure!(
            hd == 512 && nh % 32 == 0,
            "sparse_attn_tc: hd must be 512 and heads a multiple of 32"
        );
        let args = [
            Arg::P(q),
            Arg::P(win),
            Arg::P(win_idx),
            Arg::I(w as i32),
            Arg::P(comp),
            Arg::P(comp_idx),
            Arg::I(k as i32),
            Arg::P(sinks),
            Arg::P(o),
            Arg::I(nh as i32),
            Arg::I(hd as i32),
            Arg::F(1.0 / (hd as f32).sqrt()),
        ];
        if let Some(serial) = self.serial {
            // SM70: one warp per head over the gathered rows (nh % 32 == 0 implies nh % 8 == 0). The serial kernel
            // reads bf16 rows, so a packed cache has no kernel to run on — the flag gates on arch >= 80.
            ensure!(
                !self.fp4,
                "sparse_attn_tc: packed FP4 KV needs the tensor-core kernels (arch >= 80)"
            );
            return g.launch(serial, (t as u32, (nh / 8) as u32, 1), 256, 0, &args);
        }
        let smem = smem_bytes(w + k);
        ensure!(
            smem <= MAX_SMEM,
            "sparse_attn_tc: {} rows per token do not fit in smem",
            w + k
        );
        let kern = match (self.fast, self.fp4) {
            (false, false) => self.k,
            (true, false) => self.kf,
            (false, true) => self.k_fp4,
            (true, true) => self.kf_fp4,
        };
        g.launch(
            kern,
            (t as u32, (nh / 32) as u32, 1),
            256,
            smem as u32,
            &args,
        )
    }
}
