global in=0 v_position: Vector<4, Float>;
global in=1 color: Vector<4, Float>;

global out=0 f_position: Vector<4, Float>;
global out=1 f_color: Vector<4, Float>;

global position gl_position;

// single line comment
fn vertex vertex_main() {
    let a = v_position.x;
    f_position = a * v_position * if a > 2. { 2. } else if a < 3. { 3. } else { 1. };
    f_color = color;
    gl_position = v_position;
}

/* Multi line comment */
fn fragment frag_main() {
	f_position = color;
}

fn unsound() -> Vector<4,Float> {
    return v_position
}
 