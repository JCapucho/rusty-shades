fn vertex vertex_main() {
    let a = pass_trough(0);
    a = generic_function_call(test_int);
    a = generic_function_pass_trough(test_int);

    let b = pass_trough(0.0);
    let c = generic_function_call(test_int);
    let e = generic_function_pass_trough(test_int);
    let f = generic_function_pass_trough(test_float);

    let g = generic_tuple_access((test_float, 0.0));
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

fn generic_function_call<T: Fn() -> Int>(f: T) -> Int {
    f()
}

fn generic_function_pass_trough<T, F: Fn() -> T>(f: F) -> T {
    f()
}

fn generic_tuple_access<T, W>(tuple: (T, W)) -> T {
    tuple.0
}
