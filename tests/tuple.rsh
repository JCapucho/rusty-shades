fn vertex vertex_main() {
	let a = not_tuple();
	let b = tuple();
	a = tuple_access();
}

fn not_tuple() -> (Int) {
    0
}

fn tuple() -> (Int,Float) {
    (0,3.0)
}

fn tuple_access() -> Int {
    tuple().0
}
