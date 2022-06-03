use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(salsa20_force_soft)] {
        pub(crate) mod soft;
    } else if #[cfg(all(target_arch = "aarch64", target_feature = "neon"))] {
        pub(crate) mod neon;

    } else {
        pub(crate) mod soft;
    }
}
