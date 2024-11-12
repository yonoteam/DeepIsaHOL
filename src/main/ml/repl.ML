signature REPL =
sig
  type state
  val init: theory -> state
  val curr_st: state -> Toplevel.state
  val curr_act: state -> Actions.T
  val curr_err: state -> (exn * string) option
  val apply: string -> state -> state
end;


structure Repl: REPL =
struct

(* make it a theory info? *)
type state = {
    act: Actions.T, 
    st: Toplevel.state, 
    err: (exn * string) option
  } list

fun make_step (
    a: Actions.T, 
    s: Toplevel.state, 
    e: (exn * string) option) 
  = {act= a, st=s, err= e}

fun curr_st steps = #st (hd steps)
fun curr_act steps = #act (hd steps)
fun curr_err steps = #err (hd steps)

fun init thy = 
  let
    val a = Actions.void
    val s = Toplevel.make_state (SOME thy)
    val e = NONE
  in [make_step (a, s, e)] end;

fun apply txt steps =
  let
    val st = curr_st steps;
    val thy = Toplevel.theory_of st;
    val acts = Actions.make' thy txt
    val steps' = map make_step (Actions.apply_all acts st)
  in steps' @ steps  end;



end;