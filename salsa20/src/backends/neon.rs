//! The Salsa20 core function
//!

use crate::{Block, StreamClosure, Unsigned, STATE_WORDS};
use cipher::{
    consts::{U1, U64},
    BlockSizeUser, ParBlocksSizeUser, StreamBackend,
};
use core::marker::PhantomData;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use core::arch::aarch64::*;

#[inline]
pub(crate) unsafe fn inner<R, F>(state: &mut [u32; STATE_WORDS], f: F)
where
    R: Unsigned,
    F: StreamClosure<BlockSize = U64>,
{
    // If state is
    /*
         0  1  2  3
         4  5  6  7
         8  9 10 11
        12 13 14 15

    */
    // then v becomes
    /*
        0  4  8 12
        1  5  9 13
        2  6 10 14
        3  7 11 15

    */
    // This saves two matrix transposes later on, and can be done in one instruction.
    // (Can NEON fuse loads, which would make the latter point moot?)
    let mut backend = Backend::<R> {
        v: vld4q_u32(state.as_ptr()),
        _pd: PhantomData,
    };

    f.call(&mut backend);
    [state[8], state[9]] = [
        vgetq_lane_u32(backend.v.0, 2),
        vgetq_lane_u32(backend.v.1, 2),
    ];
}

pub(crate) struct Backend<R: Unsigned> {
    v: uint32x4x4_t,
    _pd: PhantomData<R>,
}

impl<R: Unsigned> BlockSizeUser for Backend<R> {
    type BlockSize = U64;
}

impl<R: Unsigned> ParBlocksSizeUser for Backend<R> {
    type ParBlocksSize = U1;
}

impl<R: Unsigned> StreamBackend for Backend<R> {
    #[inline(always)]
    fn gen_ks_block(&mut self, block: &mut Block) {
        unsafe {
            // Run rounds
            let res = rounds::<R>(self.v);

            // Update ctr
            let mut ctr =
                vgetq_lane_u32(self.v.0, 2) as u64 | (vgetq_lane_u32(self.v.1, 2) as u64) << 32;
            ctr = ctr + 1;
            self.v.0 = vsetq_lane_u32((ctr & 0xffff_ffff) as u32, self.v.0, 2);

            self.v.1 = vsetq_lane_u32(((ctr >> 32) & 0xffff_ffff) as u32, self.v.1, 2);

            // Copy res into block
            let block_ptr = block.as_mut_ptr() as *mut u32;
            vst4q_u32(block_ptr, res);
        }
    }
}
#[inline]
unsafe fn rounds<R: Unsigned>(v: uint32x4x4_t) -> uint32x4x4_t {
    /*
        v is
            [[ 0  4  8 12]
             [ 1  5  9 13]
             [ 2  6 10 14]
             [ 3  7 11 15]]

    */

    let mut diag = v.clone();
    diag.1 = vextq_u32(diag.1, diag.1, 1);
    diag.2 = vextq_u32(diag.2, diag.2, 2);
    diag.3 = vextq_u32(diag.3, diag.3, 3);

    transpose(&mut diag);

    /*
        diag is
            [[ 0  5 10 15]
            [ 4  9 14  3]
            [ 8 13  2  7]
            [12  1  6 11]]
    */

    for _ in 0..R::USIZE {
        /*
            diag is
                [[ 0  5 10 15]
                 [ 4  9 14  3]
                 [ 8 13  2  7]
                 [12  1  6 11]]
        */

        // Column round
        qr(&mut diag);

        // shuffle
        shuffle(&mut diag);

        /*
            diag is
                [[ 0  5 10 15]
                 [ 1  6 11 12]
                 [ 2  7  8 13]
                 [ 3  4  9 14]]
        */

        // Row round
        qr(&mut diag);

        // un shuffle
        deshuffle(&mut diag);
    }

    transpose(&mut diag);
    diag.1 = vextq_u32(diag.1, diag.1, 3);
    diag.2 = vextq_u32(diag.2, diag.2, 2);
    diag.3 = vextq_u32(diag.3, diag.3, 1);

    diag.0 = vaddq_u32(diag.0, v.0);
    diag.1 = vaddq_u32(diag.1, v.1);
    diag.2 = vaddq_u32(diag.2, v.2);
    diag.3 = vaddq_u32(diag.3, v.3);

    diag
}

#[inline]
unsafe fn qr(v: &mut uint32x4x4_t) {
    let ad = vaddq_u32(v.0, v.3);
    let adr = vsliq_n_u32(vshrq_n_u32(ad, 25), ad, 7);
    v.1 = veorq_u32(v.1, adr);

    let ba = vaddq_u32(v.1, v.0);
    let bar = vsliq_n_u32(vshrq_n_u32(ba, 23), ba, 9);
    v.2 = veorq_u32(v.2, bar);

    let cb = vaddq_u32(v.2, v.1);
    let cbr = vsliq_n_u32(vshrq_n_u32(cb, 19), cb, 13);
    v.3 = veorq_u32(v.3, cbr);

    let dc = vaddq_u32(v.3, v.2);
    let dcr = vsliq_n_u32(vshrq_n_u32(dc, 14), dc, 18);
    v.0 = veorq_u32(v.0, dcr);
}

#[inline]
unsafe fn shuffle(v: &mut uint32x4x4_t) {
    let vt = v.1;
    v.1 = vextq_u32(v.3, v.3, 1);
    v.2 = vextq_u32(v.2, v.2, 2);
    v.3 = vextq_u32(vt, vt, 3);
}

#[inline]
unsafe fn deshuffle(v: &mut uint32x4x4_t) {
    let vt = v.1;
    v.1 = vextq_u32(v.3, v.3, 1);
    v.2 = vextq_u32(v.2, v.2, 2);
    v.3 = vextq_u32(vt, vt, 3);
}

#[inline]
unsafe fn transpose(v: &mut uint32x4x4_t) {
    let t0 = vtrnq_u32(v.0, v.2);
    let t1 = vtrnq_u32(v.1, v.3);

    let t2 = vzipq_u32(t0.0, t1.0);
    let t3 = vzipq_u32(t0.1, t1.1);

    (v.0, v.1) = (t2.0, t3.0);
    (v.2, v.3) = (t2.1, t3.1);
}
