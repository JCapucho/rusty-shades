global in=0 v_position: Vector<4, Float>;
global in=1 v_color: Vector<4, Float>;
 
global out=1 color: Vector<4, Float>;

global position gl_position: Vector<4, Float>;

// single line comment
fn vertex vertex_main() {
	color = v_color;
	gl_position = v_position;
}

/* Multi line comment */
fn fragment frag_main() {
	color = v_color;
}
