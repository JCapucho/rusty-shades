global in=0 v_position: Vector<4, Float>;
global in=1 color: Vector<4, Float>;

global out=0 f_position: Vector<4, Float>;
global out=1 f_color: Vector<4, Float>;

global position gl_position;

fn vertex vertex_main() {
    f_position = 1.0 * v_position;
    f_color = 1.0 * color;
    gl_position = 1.0 * v_position;
    return;
}

fn fragment frag_main() {
	f_position = 1.0 * color;
	return;
}
