global in=0 v_position: Vector<4, Float>;
global in=1 color: Vector<4, Float>;

global out=0 f_position: Vector<4, Float>;
global out=1 f_color: Vector<4, Float>;

global position gl_position;

// single line comment
fn vertex vertex_main() {
    let tmp = v_position.xy;
    f_color = v4(tmp,tmp);
    gl_position = v_position;
}

/* Multi line comment */
fn fragment frag_main() {
	f_position = color;
}

fn unsound() -> Vector<4,Float> {
    return v_position
}
 