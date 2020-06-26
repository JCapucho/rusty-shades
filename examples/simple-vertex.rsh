global in=0 a_position: Vector<4, Float>;        
global out=0 v_position: Vector<4, Float>;

global position gl_position;
    
fn vertex main() {
    v_position = a_position * a_position;
    gl_position = a_position;
}
