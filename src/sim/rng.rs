//! Random streams of the simulator, and its two exactly specified floating-point sums.
//!
//! Every stream is a generator of its own, seeded from the run seed and a stream label through
//! SHA-256, so the draws of one stream never depend on how often another stream is used. Arrivals
//! of class `i` and reconfigurations on the ordered pair `(u, v)` use the Mersenne Twister MT19937
//! seeded by [`stream_seed`]; service times of class `i` use xoshiro256** seeded through
//! SplitMix64 from [`tuple_seed`]; the bootstrap resampling and the sampling checks of the
//! experiments use xoshiro256** seeded directly with a fixed integer.

use crate::digest::sha256;

/// One element of a stream label.
#[derive(Debug, Clone)]
pub enum Label {
    Str(&'static str),
    Int(u64),
}

/// Seed of a Mersenne Twister stream: the first 16 bytes (big-endian) of the
/// SHA-256 of the compact JSON list `[seed, label, ...]`, e.g.
/// `[20270901,"arrivals",0]`.
pub fn stream_seed(seed: u64, labels: &[Label]) -> u128 {
    let mut s = String::from("[");
    s.push_str(&seed.to_string());
    for l in labels {
        s.push(',');
        match l {
            Label::Str(x) => {
                s.push('"');
                s.push_str(x);
                s.push('"');
            }
            Label::Int(x) => s.push_str(&x.to_string()),
        }
    }
    s.push(']');
    let d = sha256(s.as_bytes());
    let mut bytes = [0u8; 16];
    bytes.copy_from_slice(&d[..16]);
    u128::from_be_bytes(bytes)
}

/// Seed of a xoshiro256** stream: the first 8 bytes (big-endian) of the
/// SHA-256 of the tuple text `(seed, (label, ...))`, e.g.
/// `(20270901, ('service_stream', 0))`.
pub fn tuple_seed(seed: u64, labels: &[Label]) -> u64 {
    let mut s = format!("({}, (", seed);
    for (i, l) in labels.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        match l {
            Label::Str(x) => s.push_str(&format!("'{}'", x)),
            Label::Int(x) => s.push_str(&x.to_string()),
        }
    }
    if labels.len() == 1 {
        s.push(',');
    }
    s.push_str("))");
    let d = sha256(s.as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&d[..8]);
    u64::from_be_bytes(bytes)
}

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;

#[derive(Clone)]
enum Core {
    /// MT19937.
    Mt { mt: [u32; N], mti: usize },
    /// xoshiro256**.
    Xo { s: [u64; 4] },
}

/// A random stream on one of the two generator cores. Both cores offer the operations of
/// CPython's `random.Random` (53-bit `random`, `expovariate`, the Kinderman–Monahan
/// `normalvariate`, `getrandbits`, the rejection sampler of `randrange`). On the Mersenne Twister
/// core, seeded with `init_by_array` over the 32-bit words of the seed as `random.seed` does for
/// an integer, they reproduce CPython's sequences.
#[derive(Clone)]
pub struct Rng {
    core: Core,
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

impl Rng {
    /// xoshiro256** with its state filled by four SplitMix64 outputs from `seed`.
    pub fn xoshiro(seed: u64) -> Self {
        let mut state = seed;
        let s = [
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
            splitmix64(&mut state),
        ];
        Rng {
            core: Core::Xo { s },
        }
    }

    /// MT19937 seeded with the integer `seed` (`init_by_array` over its
    /// 32-bit words, least significant first, trailing zero words removed).
    pub fn mersenne(seed: u128) -> Self {
        let mut key = vec![
            seed as u32,
            (seed >> 32) as u32,
            (seed >> 64) as u32,
            (seed >> 96) as u32,
        ];
        while key.len() > 1 && *key.last().unwrap() == 0 {
            key.pop();
        }
        let mut rng = Rng {
            core: Core::Mt {
                mt: [0; N],
                mti: N + 1,
            },
        };
        rng.init_by_array(&key);
        rng
    }

    fn mt(&mut self) -> (&mut [u32; N], &mut usize) {
        match &mut self.core {
            Core::Mt { mt, mti } => (mt, mti),
            Core::Xo { .. } => unreachable!("Mersenne Twister state of a xoshiro stream"),
        }
    }

    /// The next 64 bits: one xoshiro256** step, or two Mersenne Twister
    /// outputs (low word first).
    pub fn next_u64(&mut self) -> u64 {
        match &mut self.core {
            Core::Xo { s } => {
                let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
                let t = s[1] << 17;
                s[2] ^= s[0];
                s[3] ^= s[1];
                s[1] ^= s[2];
                s[0] ^= s[3];
                s[2] ^= t;
                s[3] = s[3].rotate_left(45);
                result
            }
            Core::Mt { .. } => {
                let low = self.next_u32() as u64;
                let high = self.next_u32() as u64;
                low | (high << 32)
            }
        }
    }

    fn init_genrand(&mut self, s: u32) {
        let (mt, mti) = self.mt();
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        *mti = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let (mt, _) = self.mt();
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = if N > key.len() { N } else { key.len() };
        while k > 0 {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1664525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1566083941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x80000000;
    }

    /// The next 32 bits: one tempered Mersenne Twister output, or the upper
    /// half of one xoshiro256** step.
    pub fn next_u32(&mut self) -> u32 {
        if let Core::Xo { .. } = self.core {
            return (self.next_u64() >> 32) as u32;
        }
        let (mt, mti) = self.mt();
        if *mti >= N {
            let mut kk = 0;
            while kk < N - M {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
                kk += 1;
            }
            while kk < N - 1 {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
                kk += 1;
            }
            let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MATRIX_A } else { 0 };
            *mti = 0;
        }
        let mut y = mt[*mti];
        *mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680;
        y ^= (y << 15) & 0xefc60000;
        y ^= y >> 18;
        y
    }

    /// A uniform double in `[0, 1)` with 53 random bits: two Mersenne
    /// Twister outputs (27 and 26 bits), or the top 53 bits of one xoshiro
    /// step.
    pub fn random(&mut self) -> f64 {
        if let Core::Xo { .. } = self.core {
            return (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0);
        }
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }

    /// An exponential variate with rate `lambd`, `-ln(1 - U) / lambd`.
    pub fn expovariate(&mut self, lambd: f64) -> f64 {
        -(1.0 - self.random()).ln() / lambd
    }

    /// A normal variate by the Kinderman–Monahan ratio-of-uniforms method.
    pub fn normalvariate(&mut self, mu: f64, sigma: f64) -> f64 {
        const NV_MAGICCONST: f64 = 1.7155277699214135; // 4 exp(-1/2) / sqrt(2)
        loop {
            let u1 = self.random();
            let u2 = 1.0 - self.random();
            let z = NV_MAGICCONST * (u1 - 0.5) / u2;
            let zz = z * z / 4.0;
            if zz <= -u2.ln() {
                return mu + z * sigma;
            }
        }
    }

    /// `exp` of a normal variate.
    pub fn lognormvariate(&mut self, mu: f64, sigma: f64) -> f64 {
        self.normalvariate(mu, sigma).exp()
    }

    /// `k <= 64` random bits as an integer.
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        if let Core::Xo { .. } = self.core {
            return self.next_u64() >> (64 - k);
        }
        if k <= 32 {
            return (self.next_u32() >> (32 - k)) as u64;
        }
        let low = self.next_u32() as u64;
        let high = (self.next_u32() >> (64 - k)) as u64;
        low | (high << 32)
    }

    /// A uniform integer in `0..n`: `getrandbits` of the bit length of `n`,
    /// rejected until below `n`.
    pub fn randrange(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        let k = 64 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }
}

/// Exactly rounded sum of finite `values` (Shewchuk's algorithm with the
/// half-even correction of the final rounding).
pub fn fsum(values: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for &value in values {
        let mut x = value;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        partials.push(x);
    }
    let mut hi = 0.0f64;
    let n = partials.len();
    if n > 0 {
        let mut idx = n;
        idx -= 1;
        hi = partials[idx];
        let mut lo = 0.0f64;
        while idx > 0 {
            let x = hi;
            idx -= 1;
            let y = partials[idx];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if idx > 0
            && ((lo < 0.0 && partials[idx - 1] < 0.0) || (lo > 0.0 && partials[idx - 1] > 0.0))
        {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    hi
}

/// Neumaier compensated summation; the accumulated correction is added at
/// the end only when it is finite and nonzero.
pub fn compensated_sum(values: &[f64]) -> f64 {
    let mut result = 0.0f64;
    let mut c = 0.0f64;
    for &x in values {
        let t = result + x;
        if result.abs() >= x.abs() {
            c += (result - t) + x;
        } else {
            c += (x - t) + result;
        }
        result = t;
    }
    if c != 0.0 && c.is_finite() {
        result += c;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_twister_matches_the_reference_sequence() {
        // Values of `random.Random(seed).random()` in CPython.
        let mut r = Rng::mersenne(0);
        assert_eq!(r.random(), 0.8444218515250481);
        assert_eq!(r.random(), 0.7579544029403025);
        let mut r = Rng::mersenne(1);
        assert_eq!(r.random(), 0.13436424411240122);
        let mut r = Rng::mersenne((1u128 << 40) + 7);
        assert_eq!(r.random(), 0.6137037779936511);
        let mut r = Rng::mersenne((1u128 << 100) + 12345);
        assert_eq!(r.random(), 0.44487310994347073);
    }

    #[test]
    fn variates_and_randrange_match_the_reference_sequence() {
        // Values of the same calls on `random.Random(seed)` in CPython.
        let mut r = Rng::mersenne(5);
        assert_eq!(r.expovariate(2.0), 0.4876246846321968);
        assert_eq!(r.lognormvariate(0.1, 0.5), 3.0423939369919317);
        let mut r = Rng::mersenne(7);
        let draws: Vec<u64> = (0..5).map(|_| r.randrange(65)).collect();
        assert_eq!(draws, vec![41, 19, 50, 6, 9]);
        assert_eq!(r.getrandbits(40), 591937865764);
    }

    #[test]
    fn stream_seeds_are_fixed() {
        assert_eq!(
            stream_seed(101, &[Label::Str("arrivals"), Label::Int(0)]),
            89451356303952040268496370717369806255
        );
        assert_eq!(
            stream_seed(101, &[Label::Str("service"), Label::Int(0), Label::Int(17)]),
            328407406370008003637221961741550841704
        );
        assert_eq!(
            stream_seed(101, &[Label::Str("switch"), Label::Int(0), Label::Int(1)]),
            250555015491414250379483294319730652734
        );
        assert_eq!(
            tuple_seed(101, &[Label::Str("service_stream"), Label::Int(0)]),
            14038782858384307380
        );
        assert_eq!(
            tuple_seed(
                12345678901234,
                &[Label::Str("service_stream"), Label::Int(3)]
            ),
            462541205919987843
        );
        assert_eq!(
            tuple_seed(99, &[Label::Str("service"), Label::Int(1), Label::Int(0)]),
            6722938337793510462
        );
        let mut r = Rng::mersenne(stream_seed(101, &[Label::Str("arrivals"), Label::Int(0)]));
        assert_eq!(r.random(), 0.4239792616262257);
        let seed = tuple_seed(101, &[Label::Str("service_stream"), Label::Int(0)]);
        let mut r = Rng::mersenne(seed as u128);
        assert_eq!(r.expovariate(1.0), 0.011622993639159095);
    }

    #[test]
    fn compensated_sum_matches_the_reference_values() {
        assert_eq!(compensated_sum(&[0.1; 10]), 1.0);
        assert_eq!(compensated_sum(&[1e16, 1.0, 1.0, -1e16]), 2.0);
        assert_eq!(
            compensated_sum(&[1.0, 1e-16, 1e-16, 1e-16]),
            1.0000000000000002
        );
        assert_eq!(compensated_sum(&[0.3, 0.3, 0.3, 0.1]), 1.0);
        assert!(compensated_sum(&[1e308, 1e308, -1e308]).is_infinite());
        assert_eq!(compensated_sum(&[2.1, 3.7, 1e-9, 5.5, 0.25]), 11.550000001);
    }

    #[test]
    fn xoshiro_streams_are_deterministic_uniform_and_distinct_per_seed() {
        let mut a = Rng::xoshiro(42);
        let mut b = Rng::xoshiro(42);
        let mut c = Rng::xoshiro(43);
        let xa: Vec<f64> = (0..8).map(|_| a.random()).collect();
        let xb: Vec<f64> = (0..8).map(|_| b.random()).collect();
        let xc: Vec<f64> = (0..8).map(|_| c.random()).collect();
        assert_eq!(xa, xb);
        assert_ne!(xa, xc);
        assert!(xa.iter().all(|u| (0.0..1.0).contains(u)));
        let mut r = Rng::xoshiro(7);
        let mean: f64 = (0..200_000).map(|_| r.expovariate(2.0)).sum::<f64>() / 200_000.0;
        assert!((mean - 0.5).abs() < 0.01);
        let draws: Vec<u64> = (0..1000).map(|_| r.randrange(65)).collect();
        assert!(draws.iter().all(|d| *d < 65));
    }

    #[test]
    fn fsum_is_exact() {
        assert_eq!(fsum(&[0.1; 10]), 1.0);
        assert_eq!(fsum(&[1e100, 1.0, -1e100, 1e-100]), 1.0);
        assert_eq!(fsum(&[0.9846153846153847, 0.015384615384615385]), 1.0);
        assert_eq!(fsum(&[1e16, 1.0, 1.0, -1e16]), 2.0);
        assert_eq!(fsum(&[1.0, 1e-16, 1e-16, 1e-16]), 1.0000000000000002);
    }
}
