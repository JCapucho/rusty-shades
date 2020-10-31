fn vertex vertex_main() {
    let a = pass_trough(0);
    a = pain(test_int);
    a = pain2(test_int);

    let b = pass_trough(0.0);
    let c = pain(test_int);
    let e = pain2(test_int);
    let f = pain2(test_float);

    let g = gen_tuple((test_float, 0.0));
    g();
}

fn test_int() -> Int {
    0
}

fn test_float() -> Float {
    0.0
}

fn pass_trough<T>(t: T) -> T {
    t
}

fn pain<T: Fn() -> Int>(f: T) -> Int {
    f()
}

fn pain2<T, F: Fn() -> T>(f: F) -> T {
    f()
}

fn gen_tuple<T, W>(tuple: (T, W)) -> T {
    tuple.0
}

fn gen_fn_tuple<T: Fn() -> W, W>(tuple: (T, W)) -> T {
    tuple.0
}

// args: --color never check