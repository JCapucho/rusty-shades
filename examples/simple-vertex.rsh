global in=0 v_position: Vector<4, Float>;
global in=1 color: Vector<4, Float>;

global out=0 f_position: Vector<4, Float>;
global out=1 f_color: Vector<4, Float>;

global position gl_position;

const HELP: Vector<3,Uint> = if false {v3(1)} else {v3(2)};

// single line comment
fn vertex vertex_main() {
    let tmp = m4(2. * v_position)[0];
    f_color = tmp;
    gl_position = v_position;
}

/* Multi line comment */
fn fragment frag_main() {
	f_position = color;
}

fn unsound() -> Vector<4,Float> {
    return v_position
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
 