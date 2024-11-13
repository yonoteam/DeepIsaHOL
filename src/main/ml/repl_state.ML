(* Mantainers:
 *   Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
 *
 * Read-eval-print-loop state (to be called by scala-isabelle)
 *)

signature REPL_STATE =
sig
  type T
  val init: theory -> T
  val get_st: T -> Toplevel.state
  val get_act: T -> Actions.T
  val get_err: T -> (exn * string) option
  val read: string * T -> T list
  val eval: T list -> T
  val print: T -> string
  val repl: string * T -> string
end;


structure Repl_State: REPL_STATE =
struct

(* make it a theory info? *)
type T = {
    act: Actions.T, 
    st: Toplevel.state, 
    err: (exn * string) option
  }

fun make_state (
    a: Actions.T, 
    s: Toplevel.state, 
    e: (exn * string) option) 
  = {act= a, st=s, err= e}

fun get_st step = #st step
fun get_act step = #act step
fun get_err step = #err step

fun init thy = make_state(
  Actions.void,
  Toplevel.make_state (SOME thy),
  NONE);

fun read (txt, step) =
  let
    val st = get_st step;
    val thy = Toplevel.theory_of st;
    val acts = Actions.make thy txt
    val steps' = map make_state (Actions.apply_all acts st)
  in steps' end;

fun eval steps = hd steps;

fun print step = Get.user_state {break_lines=false} (get_st step);

val repl = read #> eval #> print

end;