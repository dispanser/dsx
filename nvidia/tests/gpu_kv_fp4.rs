#![cfg(feature = "cuda")]
//! Packed FP4 compressed KV rows against the bf16 cache. fake_fp4_pack must reproduce fake_fp4's
//! dequantized values and the packed layout (n/2 code bytes, then n/block scale bytes); sparse_attn_tile_fp4
//! must produce the same output as sparse_attn_tile on the same rows.
//! Benchmark: cargo test --release --features cuda --test gpu_kv_fp4 -- --ignored --nocapture
use dsx::gpu::*;
use half::bf16;
use std::time::Instant;

const H: usize = 64;
const HD: usize = 512;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unif(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

// ---- CPU reference for fake_fp4, bit for bit ----

fn e4m3_round(v: f32) -> f32 {
    if v == 0.0 {
        return 0.0;
    }
    let a = v.abs();
    let q = if a < 0.015625 {
        (a * 512.0).round() / 512.0
    } else {
        let e = a.log2().floor() as i32 + 1;
        let step = 2f32.powi(e - 4);
        (a / step).round() * step
    };
    q.min(448.0).copysign(v)
}

fn pow2_ceil(x: f32) -> f32 {
    let b = x.to_bits();
    let e = ((b >> 23) & 0xFF) as i32 - 127 + i32::from(b & 0x7FFFFF != 0);
    f32::from_bits(((e + 127) as u32) << 23)
}

fn e2m1_round(v: f32) -> f32 {
    let a = v.abs().min(6.0);
    let q: f32 = if a <= 0.25 {
        0.0
    } else if a < 0.75 {
        0.5
    } else if a <= 1.25 {
        1.0
    } else if a < 1.75 {
        1.5
    } else if a <= 2.5 {
        2.0
    } else if a < 3.5 {
        3.0
    } else if a <= 5.0 {
        4.0
    } else {
        6.0
    };
    q.copysign(v)
}

fn e4m3_byte(s: f32) -> u8 {
    if s < 0.015625 {
        return (s * 512.0).round() as u8;
    }
    let e = s.log2().floor() as i32 + 1;
    let m = s / 2f32.powi(e);
    let mut exp = e + 6;
    let mut mant = ((m * 2.0 - 1.0) * 8.0).round() as i32;
    if mant == 8 {
        mant = 0;
        exp += 1;
    }
    ((exp << 3) | mant) as u8
}

fn pow2_byte(s: f32) -> u8 {
    let e = s.log2().floor() as i32 + 1;
    (e - 1 + 127) as u8
}

fn e2m1_code(v: f32) -> u8 {
    let a = v.abs();
    let c = if a == 0.0 {
        0
    } else if a == 0.5 {
        1
    } else if a == 1.0 {
        2
    } else if a == 1.5 {
        3
    } else if a == 2.0 {
        4
    } else if a == 3.0 {
        5
    } else if a == 4.0 {
        6
    } else {
        7
    };
    c | if v.is_sign_negative() { 8 } else { 0 }
}

fn e4m3_val(b: u8) -> f32 {
    let exp = (b >> 3) as i32;
    let mant = (b & 7) as f32;
    if exp == 0 {
        mant * 2f32.powi(-9)
    } else {
        (1.0 + mant * 0.125) * 2f32.powi(exp - 7)
    }
}

fn pow2_val(b: u8) -> f32 {
    2f32.powi(b as i32 - 127)
}

fn e2m1_val(c: u8) -> f32 {
    let t = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = t[(c & 7) as usize];
    if c & 8 != 0 {
        -v
    } else {
        v
    }
}

fn block_scale(x: &[f32], e4m3s: bool) -> f32 {
    let amax = x.iter().fold(0f32, |a, &v| a.max(v.abs()));
    if e4m3s {
        e4m3_round(amax.max(6.0 * 0.001953125) / 6.0)
    } else {
        pow2_ceil(amax.max(6.0 * 1.1754944e-38) / 6.0)
    }
}

/// fake_fp4's in-place dequantized values, per block.
fn quantize_row(x: &[f32], block: usize, e4m3s: bool) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for b in 0..x.len() / block {
        let blk = &x[b * block..(b + 1) * block];
        let s = block_scale(blk, e4m3s);
        for (j, &v) in blk.iter().enumerate() {
            out[b * block + j] = e2m1_round((v / s).clamp(-6.0, 6.0)) * s;
        }
    }
    out
}

/// The packed row: n/2 code bytes, then n/block scale bytes.
fn pack_row(x: &[f32], block: usize, e4m3s: bool) -> Vec<u8> {
    let n = x.len();
    let mut out = vec![0u8; n / 2 + n / block];
    for b in 0..n / block {
        let blk = &x[b * block..(b + 1) * block];
        let s = block_scale(blk, e4m3s);
        out[n / 2 + b] = if e4m3s { e4m3_byte(s) } else { pow2_byte(s) };
        for (j, &v) in blk.iter().enumerate() {
            let code = e2m1_code(e2m1_round((v / s).clamp(-6.0, 6.0)));
            let i = b * block + j;
            out[i / 2] |= if i.is_multiple_of(2) { code } else { code << 4 };
        }
    }
    out
}

fn unpack_row(p: &[u8], n: usize, block: usize, e4m3s: bool) -> Vec<f32> {
    let mut out = vec![0f32; n];
    for b in 0..n / block {
        let s = if e4m3s { e4m3_val(p[n / 2 + b]) } else { pow2_val(p[n / 2 + b]) };
        for j in 0..block {
            let i = b * block + j;
            let byte = p[i / 2];
            let code = if i.is_multiple_of(2) { byte & 0xF } else { byte >> 4 };
            out[i] = e2m1_val(code) * s;
        }
    }
    out
}

fn pack_rows(g: &Gpu, rows: &[f32], n: usize, block: usize, e4m3s: bool) -> Buf {
    let stride = n / 2 + n / block;
    let dst = g.alloc(rows.len() / n * stride).unwrap();
    let k = g.kernel("fake_fp4_pack").unwrap();
    for i in 0..rows.len() / n {
        let x = g.upload(&rows[i * n..(i + 1) * n]).unwrap();
        let args = [Arg::P(x.ptr), Arg::I(n as i32), Arg::I(block as i32), Arg::I(e4m3s as i32), Arg::P(dst.ptr + (i * stride) as u64)];
        g.launch(k, ((n as u32 / 32).div_ceil(8), 1, 1), 256, 0, &args).unwrap();
    }
    dst
}

#[test]
fn pack_matches_fake_fp4() {
    let g = Gpu::new(0).unwrap();
    let k = g.kernel("fake_fp4_pack").unwrap();
    let mut r = Rng(0xdeadbeef);
    for (n, block, e4m3s) in [(HD, 16usize, true), (128, 32, false), (HD, 16, false), (128, 32, true)] {
        for round in 0..8 {
            // a mix of magnitudes so the scales and the E2M1 grid are both exercised
            let x: Vec<f32> = (0..n).map(|_| r.unif() * 2f32.powi(round - 3)).collect();
            let want = pack_row(&x, block, e4m3s);
            let gx = g.upload(&x).unwrap();
            let gp = g.alloc(want.len()).unwrap();
            let args = [Arg::P(gx.ptr), Arg::I(n as i32), Arg::I(block as i32), Arg::I(e4m3s as i32), Arg::P(gp.ptr)];
            g.launch(k, ((n as u32 / 32).div_ceil(8), 1, 1), 256, 0, &args).unwrap();
            let got = g.download::<u8>(gp.ptr, want.len()).unwrap();
            assert_eq!(got, want, "packed bytes differ (n={n} block={block} e4m3s={e4m3s} round={round})");
            // in-place values == fake_fp4's
            let refq = quantize_row(&x, block, e4m3s);
            let gotq = g.download::<f32>(gx.ptr, n).unwrap();
            assert_eq!(gotq, refq, "fake_fp4 values differ (n={n} block={block} round={round})");
            // unpacking the packed bytes == the same values, after the bf16 store
            let unp: Vec<bf16> = unpack_row(&got, n, block, e4m3s).iter().map(|&v| bf16::from_f32(v)).collect();
            let want_bf: Vec<bf16> = refq.iter().map(|&v| bf16::from_f32(v)).collect();
            assert_eq!(unp, want_bf, "unpack differs (n={n} block={block} round={round})");
        }
    }
}

/// The decode shape: T=1, W=128, K=512, H=64, HD=512.
#[test]
fn sparse_attn_fp4_matches() {
    let g = Gpu::new(0).unwrap();
    let (tile, tile_fp4) = (g.kernel("sparse_attn_tile").unwrap(), g.kernel("sparse_attn_tile_fp4").unwrap());
    let merge = g.kernel("sparse_attn_merge").unwrap();
    let mut r = Rng(0x51ab1e);
    let scale = 1.0 / (HD as f32).sqrt();
    let (nc, nw) = (600usize, 300usize);
    let comp_f32: Vec<f32> = (0..nc * HD).map(|_| r.unif() * 2.0).collect();
    let comp_bf: Vec<bf16> = (0..nc)
        .flat_map(|i| quantize_row(&comp_f32[i * HD..(i + 1) * HD], 16, true))
        .map(bf16::from_f32)
        .collect();
    let packed = pack_rows(&g, &comp_f32, HD, 16, true);
    // the packed rows must unpack to the reference bf16 rows
    let stride = HD / 2 + HD / 16;
    let got = g.download::<u8>(packed.ptr, nc * stride).unwrap();
    for i in 0..nc {
        let unp: Vec<bf16> = unpack_row(&got[i * stride..(i + 1) * stride], HD, 16, true).iter().map(|&v| bf16::from_f32(v)).collect();
        assert_eq!(unp, comp_bf[i * HD..(i + 1) * HD], "row {i} unpack differs");
    }
    let win: Vec<bf16> = (0..nw * HD).map(|_| bf16::from_f32(r.unif() * 2.0)).collect();
    let bq0 = g.upload(&win).unwrap();
    let _ = bq0;
    for (t, w, k) in [(1usize, 128usize, 512usize), (4, 128, 512), (3, 100, 37)] {
        let pick = |r: &mut Rng, n: usize| if r.below(10) == 0 { -1 } else { r.below(n) as i32 };
        let win_idx: Vec<i32> = (0..t * w).map(|_| pick(&mut r, nw)).collect();
        let comp_idx: Vec<i32> = (0..t * k).map(|_| pick(&mut r, nc)).collect();
        let q: Vec<f32> = (0..t * H * HD).map(|_| r.unif() * 2.0).collect();
        let sinks: Vec<f32> = (0..H).map(|_| r.unif() * 3.0).collect();
        let bq = g.upload(&q).unwrap();
        let bw = g.upload(&win).unwrap();
        let bc = g.upload(&comp_bf).unwrap();
        let bwi = g.upload(&win_idx).unwrap();
        let bci = g.upload(&comp_idx).unwrap();
        let bsk = g.upload(&sinks).unwrap();
        let nz = (w + k).div_ceil(32);
        let (pa, pb) = (g.alloc(t * nz * H * (HD + 2) * 4).unwrap(), g.alloc(t * nz * H * (HD + 2) * 4).unwrap());
        let (oa, ob) = (g.alloc(t * H * HD * 4).unwrap(), g.alloc(t * H * HD * 4).unwrap());
        let args = |comp: u64, part: u64| {
            [
                Arg::P(bq.ptr),
                Arg::P(bw.ptr),
                Arg::P(bwi.ptr),
                Arg::I(w as i32),
                Arg::P(comp),
                Arg::P(bci.ptr),
                Arg::I(k as i32),
                Arg::P(part),
                Arg::I(H as i32),
                Arg::I(HD as i32),
                Arg::F(scale),
            ]
        };
        let grid = (t as u32, (H / 16) as u32, nz as u32);
        g.launch(tile, grid, 256, 0, &args(bc.ptr, pa.ptr)).unwrap();
        g.launch(tile_fp4, grid, 256, 0, &args(packed.ptr, pb.ptr)).unwrap();
        let a = g.download::<f32>(pa.ptr, t * nz * H * (HD + 2)).unwrap();
        let b = g.download::<f32>(pb.ptr, t * nz * H * (HD + 2)).unwrap();
        assert_eq!(a, b, "part differs (T={t} W={w} K={k})");
        for (p, o) in [(pa.ptr, oa.ptr), (pb.ptr, ob.ptr)] {
            let margs = [Arg::P(p), Arg::I(nz as i32), Arg::P(bsk.ptr), Arg::P(o), Arg::I(H as i32), Arg::I(HD as i32)];
            g.launch(merge, (t as u32, H as u32, 1), 256, 0, &margs).unwrap();
        }
        let oa_v = g.download::<f32>(oa.ptr, t * H * HD).unwrap();
        let ob_v = g.download::<f32>(ob.ptr, t * H * HD).unwrap();
        assert_eq!(oa_v, ob_v, "merged output differs (T={t} W={w} K={k})");
        assert!(oa_v.iter().all(|v| v.is_finite()));
    }
}

/// The prefill shape (T > 8): sparse_attn_tc against sparse_attn_tc_fp4 on the same rows.
#[test]
fn sparse_attn_tc_fp4_matches() {
    let g = Gpu::new(0).unwrap();
    let mut tc = dsx::attn::SparseAttnTc::new(&g).unwrap();
    let mut r = Rng(0x7e57);
    let (nc, nw) = (600usize, 300usize);
    let comp_f32: Vec<f32> = (0..nc * HD).map(|_| r.unif() * 2.0).collect();
    let comp_bf: Vec<bf16> = (0..nc)
        .flat_map(|i| quantize_row(&comp_f32[i * HD..(i + 1) * HD], 16, true))
        .map(bf16::from_f32)
        .collect();
    let packed = pack_rows(&g, &comp_f32, HD, 16, true);
    let win: Vec<bf16> = (0..nw * HD).map(|_| bf16::from_f32(r.unif() * 2.0)).collect();
    for (t, w, k) in [(6usize, 128usize, 512usize), (4, 128, 512)] {
        let pick = |r: &mut Rng, n: usize| if r.below(10) == 0 { -1 } else { r.below(n) as i32 };
        let win_idx: Vec<i32> = (0..t * w).map(|_| pick(&mut r, nw)).collect();
        let comp_idx: Vec<i32> = (0..t * k).map(|_| pick(&mut r, nc)).collect();
        let q: Vec<f32> = (0..t * H * HD).map(|_| r.unif() * 2.0).collect();
        let sinks: Vec<f32> = (0..H).map(|_| r.unif() * 3.0).collect();
        let (bq, bw, bc) = (g.upload(&q).unwrap(), g.upload(&win).unwrap(), g.upload(&comp_bf).unwrap());
        let (bwi, bci, bsk) = (g.upload(&win_idx).unwrap(), g.upload(&comp_idx).unwrap(), g.upload(&sinks).unwrap());
        let o = g.alloc(t * H * HD * 4).unwrap();
        tc.fast = false;
        tc.fp4 = false;
        tc.launch(&g, bq.ptr, bw.ptr, bwi.ptr, w, bc.ptr, bci.ptr, k, bsk.ptr, o.ptr, t, H, HD).unwrap();
        let a = g.download::<f32>(o.ptr, t * H * HD).unwrap();
        g.memset(o.ptr, t * H * HD * 4).unwrap();
        tc.fp4 = true;
        tc.launch(&g, bq.ptr, bw.ptr, bwi.ptr, w, packed.ptr, bci.ptr, k, bsk.ptr, o.ptr, t, H, HD).unwrap();
        let b = g.download::<f32>(o.ptr, t * H * HD).unwrap();
        assert_eq!(a, b, "tc fp4 differs (T={t} W={w} K={k})");
    }
}

/// The indexer keys: indexer_score against indexer_score_fp4 on the same rows.
#[test]
fn indexer_score_fp4_matches() {
    let g = Gpu::new(0).unwrap();
    let (kb, kp) = (g.kernel("indexer_score").unwrap(), g.kernel("indexer_score_fp4").unwrap());
    let mut r = Rng(0x1d3e);
    let (id, ih, n, t) = (128usize, 32usize, 1000usize, 3usize);
    let keys_f32: Vec<f32> = (0..n * id).map(|_| r.unif() * 2.0).collect();
    let keys_bf: Vec<bf16> = (0..n)
        .flat_map(|i| quantize_row(&keys_f32[i * id..(i + 1) * id], 32, false))
        .map(bf16::from_f32)
        .collect();
    let packed = pack_rows(&g, &keys_f32, id, 32, false);
    let q: Vec<f32> = (0..t * ih * id).map(|_| r.unif() * 2.0).collect();
    let w: Vec<f32> = (0..t * ih).map(|_| r.unif()).collect();
    let nvis: Vec<i32> = (0..t).map(|i| (n - i * 100) as i32).collect();
    let (bq, bk, bw) = (g.upload(&q).unwrap(), g.upload(&keys_bf).unwrap(), g.upload(&w).unwrap());
    let bnv = g.upload(&nvis).unwrap();
    let (s1, s2) = (g.alloc(t * n * 4).unwrap(), g.alloc(t * n * 4).unwrap());
    let grid = ((n as u32).div_ceil(8), t as u32, 1);
    let args = |k: u64, s: u64| {
        [Arg::P(bq.ptr), Arg::P(k), Arg::P(bw.ptr), Arg::P(s), Arg::P(bnv.ptr), Arg::I(n as i32), Arg::I(ih as i32), Arg::I(id as i32)]
    };
    g.launch(kb, grid, 256, 0, &args(bk.ptr, s1.ptr)).unwrap();
    g.launch(kp, grid, 256, 0, &args(packed.ptr, s2.ptr)).unwrap();
    let a = g.download::<f32>(s1.ptr, t * n).unwrap();
    let b = g.download::<f32>(s2.ptr, t * n).unwrap();
    assert_eq!(a, b, "indexer score differs");
}

fn time(f: &mut dyn FnMut(), g: &Gpu, n: usize) -> f64 {
    f();
    g.sync().unwrap();
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    g.sync().unwrap();
    t.elapsed().as_secs_f64() / n as f64 * 1e6
}

/// tile + merge at the decode shape, bf16 cache against the packed one, for an L2-resident and a DRAM pool.
#[test]
#[ignore]
fn sparse_attn_fp4_bench() {
    let g = Gpu::new(0).unwrap();
    let mut r = Rng(12345);
    let (t, w, k, nw) = (1usize, 128usize, 512usize, 300usize);
    let win: Vec<bf16> = (0..nw * HD).map(|_| bf16::from_f32(r.unif() * 2.0)).collect();
    let win_idx: Vec<i32> = (0..t * w).map(|_| r.below(nw) as i32).collect();
    let q: Vec<f32> = (0..t * H * HD).map(|_| r.unif() * 2.0).collect();
    let sinks: Vec<f32> = (0..H).map(|_| r.unif() * 3.0).collect();
    let (bq, bw) = (g.upload(&q).unwrap(), g.upload(&win).unwrap());
    let (bwi, bsk) = (g.upload(&win_idx).unwrap(), g.upload(&sinks).unwrap());
    let nz = (w + k).div_ceil(32);
    let part = g.alloc(t * nz * H * (HD + 2) * 4).unwrap();
    let o = g.alloc(t * H * HD * 4).unwrap();
    let (tile, tile_fp4, merge) = (
        g.kernel("sparse_attn_tile").unwrap(),
        g.kernel("sparse_attn_tile_fp4").unwrap(),
        g.kernel("sparse_attn_merge").unwrap(),
    );
    let scale = 1.0 / (HD as f32).sqrt();
    eprintln!("{} ({} SMs), T={t} W={w} K={k}", g.name, g.sms);
    for nc in [2000usize, 32768] {
        let comp_f32: Vec<f32> = (0..nc * HD).map(|_| r.unif() * 2.0).collect();
        let comp_bf: Vec<bf16> = (0..nc)
            .flat_map(|i| quantize_row(&comp_f32[i * HD..(i + 1) * HD], 16, true))
            .map(bf16::from_f32)
            .collect();
        let packed = pack_rows(&g, &comp_f32, HD, 16, true);
        let comp_idx: Vec<i32> = (0..t * k).map(|_| r.below(nc) as i32).collect();
        let (bc, bci) = (g.upload(&comp_bf).unwrap(), g.upload(&comp_idx).unwrap());
        let args = |comp: u64| {
            [
                Arg::P(bq.ptr),
                Arg::P(bw.ptr),
                Arg::P(bwi.ptr),
                Arg::I(w as i32),
                Arg::P(comp),
                Arg::P(bci.ptr),
                Arg::I(k as i32),
                Arg::P(part.ptr),
                Arg::I(H as i32),
                Arg::I(HD as i32),
                Arg::F(scale),
            ]
        };
        let grid = (t as u32, (H / 16) as u32, nz as u32);
        let margs = [Arg::P(part.ptr), Arg::I(nz as i32), Arg::P(bsk.ptr), Arg::P(o.ptr), Arg::I(H as i32), Arg::I(HD as i32)];
        let run = |kern: Kernel, comp: u64| {
            g.launch(kern, grid, 256, 0, &args(comp)).unwrap();
            g.launch(merge, (t as u32, H as u32, 1), 256, 0, &margs).unwrap();
        };
        let t_bf = time(&mut || run(tile, bc.ptr), &g, 500);
        let t_fp4 = time(&mut || run(tile_fp4, packed.ptr), &g, 500);
        let t_tile = time(&mut || g.launch(tile, grid, 256, 0, &args(bc.ptr)).unwrap(), &g, 500);
        let t_tile_fp4 = time(&mut || g.launch(tile_fp4, grid, 256, 0, &args(packed.ptr)).unwrap(), &g, 500);
        let mb = nc * HD * 2 / 1_000_000;
        eprintln!("pool {nc} rows ({mb} MB bf16, {} MB fp4):", nc * (HD / 2 + HD / 16) / 1_000_000);
        eprintln!("  tile+merge bf16 {t_bf:7.1} us   fp4 {t_fp4:7.1} us   {:.1}%", (t_bf - t_fp4) / t_bf * 100.0);
        eprintln!("  tile       bf16 {t_tile:7.1} us   fp4 {t_tile_fp4:7.1} us   {:.1}%", (t_tile - t_tile_fp4) / t_tile * 100.0);
    }
}
