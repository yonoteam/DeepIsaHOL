(* Mantainers:
 *   Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
 *
 * Read-eval-print-loop state (to be called by scala-isabelle)
 *)

signature REPL_STATE =
sig
  type T
  exception E_Repl_State of T
  val init: theory -> T
  val get_st: T -> Toplevel.state
  val get_act: T -> Actions.T
  val get_err: T -> (exn * string) option
  val read: string * T -> T
  val eval: T * T -> T
  val print: T -> string
  val repl: string * T -> string
  val test_str: string
end;


structure Repl_State: REPL_STATE =
struct

(* TODO: improve *)
(* State working as a stack *)
type T = {
    act: Actions.T, 
    st: Toplevel.state, 
    err: (exn * string) option
  } list

(* Needed for scala-isabelle *)
exception E_Repl_State of T

fun make_one (
    a: Actions.T, 
    s: Toplevel.state, 
    e: (exn * string) option) 
  = {act= a, st=s, err= e}

fun get_st state = #st (hd state)
fun get_act state = #act (hd state)
fun get_err state = #err (hd state)

fun init thy = [make_one(
  Actions.void,
  Toplevel.make_state (SOME thy),
  NONE)];

fun read (txt, state) =
  let
    val st = get_st state;
    val thy = Toplevel.theory_of st;
    val acts = Actions.make thy txt
    val new_sts = map make_one (Actions.apply_all acts st)
  in new_sts end;

fun eval (new_sts, state) = new_sts @ state;

fun print state = Get.user_state {break_lines=false} (get_st state);

fun repl (txt, state) = read (txt, state) |> (fn states' => eval (states', state)) |> print

val test_str = "message passed"

end;