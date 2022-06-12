//! The ChaCha20 core function. Defined in RFC 8439 Section 2.3.
//!
//! <https://tools.ietf.org/html/rfc8439#section-2.3>
//!
//! Port of x86 sse2 backend to NEON instructions

use crate::{Block, StreamClosure, Unsigned, STATE_WORDS};
use cipher::{
    consts::{U4, U64},
    BlockSizeUser, ParBlocksSizeUser, StreamBackend,
};
use core::marker::PhantomData;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use core::arch::aarch64::*;

const PAR_BLOCKS: usize = 4;

#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn inner<R, F>(state: &mut [u32; STATE_WORDS], f: F)
where
    R: Unsigned,
    F: StreamClosure<BlockSize = U64>,
{
    let state_ptr = state.as_ptr();
    let mut v = [
        vld1q_dup_u32(state_ptr.add(0x0)),
        vld1q_dup_u32(state_ptr.add(0x1)),
        vld1q_dup_u32(state_ptr.add(0x2)),
        vld1q_dup_u32(state_ptr.add(0x3)),
        vld1q_dup_u32(state_ptr.add(0x4)),
        vld1q_dup_u32(state_ptr.add(0x5)),
        vld1q_dup_u32(state_ptr.add(0x6)),
        vld1q_dup_u32(state_ptr.add(0x7)),
        vld1q_dup_u32(state_ptr.add(0x8)),
        vld1q_dup_u32(state_ptr.add(0x9)),
        vld1q_dup_u32(state_ptr.add(0xa)),
        vld1q_dup_u32(state_ptr.add(0xb)),
        vld1q_dup_u32(state_ptr.add(0xc)),
        vld1q_dup_u32(state_ptr.add(0xd)),
        vld1q_dup_u32(state_ptr.add(0xe)),
        vld1q_dup_u32(state_ptr.add(0xf)),
    ];

    // Incoming state words don't have counter set properly
    v[12] = vaddq_u32(v[12], vld1q_u32((&[0u32, 1, 2, 3]) as *const u32));

    let mut backend = Backend::<R> {
        v,
        _pd: PhantomData,
    };

    f.call(&mut backend);
    state[12] = vgetq_lane_u32(backend.v[12], 0) as u32;
}

pub(crate) struct Backend<R: Unsigned> {
    v: [uint32x4_t; STATE_WORDS],
    _pd: PhantomData<R>,
}

impl<R: Unsigned> BlockSizeUser for Backend<R> {
    type BlockSize = U64;
}

impl<R: Unsigned> ParBlocksSizeUser for Backend<R> {
    type ParBlocksSize = U4;
}

impl<R: Unsigned> StreamBackend for Backend<R> {
    #[inline(always)]
    fn gen_ks_block(&mut self, block: &mut Block) {
        unsafe {
            // Run rounds
            let res = rounds::<R>(&self.v);

            // Increment counter
            self.v[12] = vaddq_u32(self.v[12], vld1q_dup_u32(&1));

            let block_ptr = block.as_mut_ptr() as *mut u32;
            for i in 0..STATE_WORDS {
                // Which lane we store doesn't matter
                vst1q_lane_u32::<0>(block_ptr.add(i), res[i]);
            }
        }
    }

    #[inline(always)]
    fn gen_par_ks_blocks(&mut self, blocks: &mut cipher::ParBlocks<Self>) {
        unsafe {
            // Run rounds
            let res = rounds::<R>(&self.v);

            let pb = PAR_BLOCKS as u32;
            self.v[12] = vaddq_u32(self.v[12], vld1q_dup_u32(&pb));

            let b0 = blocks[0].as_ptr() as *mut u32;
            let b1 = blocks[1].as_ptr() as *mut u32;
            let b2 = blocks[2].as_ptr() as *mut u32;
            let b3 = blocks[3].as_ptr() as *mut u32;

            for i in 0..STATE_WORDS {
                vst1q_lane_u32::<0>(b0.add(i), res[i]);
                vst1q_lane_u32::<1>(b1.add(i), res[i]);
                vst1q_lane_u32::<2>(b2.add(i), res[i]);
                vst1q_lane_u32::<3>(b3.add(i), res[i]);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn rounds<R: Unsigned>(v: &[uint32x4_t; STATE_WORDS]) -> [uint32x4_t; STATE_WORDS] {
    let mut res = *v;
    for _ in 0..R::USIZE {
        double_round(&mut res);
    }

    for i in 0..STATE_WORDS {
        res[i] = vaddq_u32(v[i], res[i]);
    }
    res
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn double_round(v: &mut [uint32x4_t; STATE_WORDS]) {
    const VALS: [[usize; 4]; 8] = [
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
        [0, 5, 10, 15],
        [1, 6, 11, 12],
        [2, 7, 8, 13],
        [3, 4, 9, 14],
    ];
    for i in 0..8 {
        qr(v, VALS[i]);
    }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn qr(v: &mut [uint32x4_t; STATE_WORDS], [a, b, c, d]: [usize; 4]) {
    // a += b; d ^= a; d <<<= (16, 16, 16, 16);
    v[a] = vaddq_u32(v[a], v[b]);
    v[d] = veorq_u32(v[d], v[a]);
    v[d] = vsliq_n_u32(vshrq_n_u32(v[d], 16), v[d], 16);

    // c += d; b ^= c; b <<<= (12, 12, 12, 12);
    v[c] = vaddq_u32(v[c], v[d]);
    v[b] = veorq_u32(v[b], v[c]);
    v[b] = vsliq_n_u32(vshrq_n_u32(v[b], 20), v[b], 12);

    // a += b; d ^= a; d <<<= (8, 8, 8, 8);
    v[a] = vaddq_u32(v[a], v[b]);
    v[d] = veorq_u32(v[d], v[a]);
    v[d] = vsliq_n_u32(vshrq_n_u32(v[d], 24), v[d], 8);

    // c += d; b ^= c; b <<<= (7, 7, 7, 7);
    v[c] = vaddq_u32(v[c], v[d]);
    v[b] = veorq_u32(v[b], v[c]);
    v[b] = vsliq_n_u32(vshrq_n_u32(v[b], 25), v[b], 7);
}
