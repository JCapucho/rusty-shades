global in=0 v_position: Vector<4, Float>;
global in=1 color: Vector<4, Float>;

global out=0 f_position: Vector<4, Float>;
global out=1 f_color: Vector<4, Float>;

global position gl_position: Vector<4, Float>;

const TRUE_HELP: Vector<3,Uint> = v3(1);
const FALSE_HELP: Vector<3,Uint> = v3(2);
const HELP: Vector<3,Uint> = if false {TRUE_HELP} else {FALSE_HELP};

// single line comment
fn vertex vertex_main() {
    let a = pass_trough(0);
    let b = pass_trough(0.0);
    a = test_pain();
    a = pain2(tuple_access);
    let c: (Int, Float) = pain2(tuple);

    let tmp = m4(2. * v_position)[0];
    f_color = tmp;
    gl_position = v_position;
}

/* Multi line comment */
fn fragment frag_main() {
	f_position = color;
}

fn unsound() -> Vector<4,Float> {
    v4(1.)
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

struct Test(Int,Float);

fn tuple_struct(strct: Test) -> Int {
    strct.0
}

fn test_pass_trough(t: Int) -> Int {
    pass_trough(t)
}

fn test_pain() -> Int {
    pain(tuple_access)
}

// fn test_pain_pass_trough() -> Int {
//     pass_trough(pain(tuple_access))
// }

fn pass_trough<T>(t: T) -> T {
    t
}

fn pain<T: Fn() -> Int>(f: T) -> Int {
    f()
}

fn pain2<T, F: Fn() -> T>(f: F) -> T {
    f()
}
