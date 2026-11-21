// The browser WASI host exposes a nonblocking stop byte on fd 4.
pub fn requested() -> bool {
    #[repr(C)]
    struct Iovec {
        buf: *mut u8,
        len: usize,
    }
    #[link(wasm_import_module = "wasi_snapshot_preview1")]
    unsafe extern "C" {
        #[link_name = "fd_read"]
        fn stop_fd_read(fd: u32, iovs: *const Iovec, len: usize, nread: *mut usize) -> u32;
    }

    let mut stop = 0u8;
    let iov = Iovec { buf: &mut stop, len: 1 };
    let mut nread = 0usize;
    unsafe { stop_fd_read(4, &iov, 1, &mut nread) == 0 && nread == 1 && stop != 0 }
}
