const TRUE_HELP: Vector<3,Uint> = v3(1);
const FALSE_HELP: Vector<3,Uint> = v3(2);
const HELP: Vector<3,Uint> = if false {TRUE_HELP} else {FALSE_HELP};
