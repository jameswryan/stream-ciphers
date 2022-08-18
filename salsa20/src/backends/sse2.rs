use crate::{Block, StreamClosure, Unsigned, STATE_WORDS};
use cipher::{
    consts::{U1, U64},
    BlockSizeUser, ParBlocksSizeUser, StreamBackend,
};
use core::marker::PhantomData;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse2")]
pub(crate) unsafe fn inner<R, F>(state: &mut [u32; STATE_WORDS], f: F)
where
    R: Unsigned,
    F: StreamClosure<BlockSize = U64>,
{
    let state_ptr = state.as_ptr() as *const __m128i;
    let mut backend = Backend::<R> {
        v: [
            _mm_loadu_si128(state_ptr.add(0)),
            _mm_loadu_si128(state_ptr.add(1)),
            _mm_loadu_si128(state_ptr.add(2)),
            _mm_loadu_si128(state_ptr.add(3)),
        ],
        _pd: PhantomData,
    };

    f.call(&mut backend);

    let c = _mm_cvtsi128_si64(backend.v[2]) as u64;
    state[8] = (c & 0xffff_ffff) as u32;
    state[9] = ((c >> 32) & 0xffff_ffff) as u32;
}

struct Backend<R: Unsigned> {
    v: [__m128i; 4],
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
            let res = rounds::<R>(&self.v);

            // increment counter
            self.v[2] = _mm_add_epi64(self.v[2], _mm_set_epi64x(0, 1));

            let block_ptr = block.as_mut_ptr() as *mut __m128i;
            for i in 0..4 {
                _mm_storeu_si128(block_ptr.add(i), res[i]);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn rounds<R: Unsigned>(v: &[__m128i; 4]) -> [__m128i; 4] {
    let mut res = *v;
    // res is
    /*
        [[ 0  1  2  3]
         [ 4  5  6  7]
         [ 8  9 10 11]
         [12 13 14 15]]
    */
    // But to vectorize the quarter round we want
    /*
        [[ 0  5 10 15]
         [ 4  9 14  3]
         [ 8 13  2  7]
         [12  1  6 11]]
    */
    // To get this we do a transpose, followed by a shuffle, followed by a transpose.
    res = transpose(res);
    res = [
        _mm_shuffle_epi32(res[0], 0b11_10_01_00),
        _mm_shuffle_epi32(res[1], 0b00_11_10_01),
        _mm_shuffle_epi32(res[2], 0b01_00_11_10),
        _mm_shuffle_epi32(res[3], 0b10_01_00_11),
    ];
    res = transpose(res);

    for _ in 0..R::USIZE {
        // res is
        /*
            [[ 0  5 10 15]
             [ 4  9 14  3]
             [ 8 13  2  7]
             [12  1  6 11]]
        */

        // Column round
        res = qr(&res);

        // shuffle
        res = shuffle(&res);

        // res is
        /*
            [[ 0  5 10 15]
             [ 1  6 11 12]
             [ 2  7  8 13]
             [ 3  4  9 14]]
        */

        // Row round
        res = qr(&res);

        // un shuffle
        res = shuffle(&res);
    }

    // Undo shuffle from before rounds
    res = transpose(res);
    res = [
        _mm_shuffle_epi32(res[0], 0b11_10_01_00),
        _mm_shuffle_epi32(res[1], 0b10_01_00_11),
        _mm_shuffle_epi32(res[2], 0b01_00_11_10),
        _mm_shuffle_epi32(res[3], 0b00_11_10_01),
    ];
    res = transpose(res);

    for i in 0..4 {
        res[i] = _mm_add_epi32(res[i], v[i]);
    }

    res
}

/// Salsa 20 quarter round function
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn qr([a, b, c, d]: &[__m128i; 4]) -> [__m128i; 4] {
    let mut ret = [*a, *b, *c, *d];
    // b ^= (a+d) <<< 7
    let ad = _mm_add_epi32(ret[0], ret[3]);
    let adr = _mm_xor_si128(_mm_slli_epi32(ad, 7), _mm_srli_epi32(ad, 25));
    ret[1] = _mm_xor_si128(adr, ret[1]);

    // c ^= (b+a) <<< 9
    let ba = _mm_add_epi32(ret[1], ret[0]);
    let bar = _mm_xor_si128(_mm_slli_epi32(ba, 9), _mm_srli_epi32(ba, 23));
    ret[2] = _mm_xor_si128(bar, ret[2]);

    // d ^= (c+b) <<< 13
    let cb = _mm_add_epi32(ret[2], ret[1]);
    let cbr = _mm_xor_si128(_mm_slli_epi32(cb, 13), _mm_srli_epi32(cb, 19));
    ret[3] = _mm_xor_si128(cbr, ret[3]);

    // a ^= (d+c) <<< 18
    let dc = _mm_add_epi32(ret[3], ret[2]);
    let dcr = _mm_xor_si128(_mm_slli_epi32(dc, 18), _mm_srli_epi32(dc, 14));
    ret[0] = _mm_xor_si128(dcr, ret[0]);

    ret
}

#[inline]
#[target_feature(enable = "sse2")]
unsafe fn shuffle(res: &[__m128i; 4]) -> [__m128i; 4] {
    [
        _mm_shuffle_epi32(res[0], 0b11_10_01_00),
        _mm_shuffle_epi32(res[3], 0b00_11_10_01),
        _mm_shuffle_epi32(res[2], 0b01_00_11_10),
        _mm_shuffle_epi32(res[1], 0b10_01_00_11),
    ]
}

/// 4x4 matrix of 32bit values stored in array
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn transpose(mati: [__m128i; 4]) -> [__m128i; 4] {
    // Use _MM_TRANSPOSE4_PS but do nasty typecasting
    // ugly cast
    let [mut m0, mut m1, mut m2, mut m3] = [
        _mm_castsi128_ps(mati[0]),
        _mm_castsi128_ps(mati[1]),
        _mm_castsi128_ps(mati[2]),
        _mm_castsi128_ps(mati[3]),
    ];
    _MM_TRANSPOSE4_PS(&mut m0, &mut m1, &mut m2, &mut m3);
    // another ugly cast
    [
        _mm_castps_si128(m0),
        _mm_castps_si128(m1),
        _mm_castps_si128(m2),
        _mm_castps_si128(m3),
    ]
}
