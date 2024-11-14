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
  val get_err: T -> string
  val size: T -> int
  val read: string * T -> T
  val eval: T * T -> T
  val print: T -> string
  val read_eval: string * T -> T
  val repl: string * T -> string
end;


structure Repl_State: REPL_STATE =
struct

(* TODO: improve *)
(* State working as a stack *)
type T = {
    act: Actions.T, 
    st: Toplevel.state, 
    err: string
  } list

(* Needed for scala-isabelle *)
exception E_Repl_State of T

fun make_one (
    a: Actions.T, 
    s: Toplevel.state, 
    e: string) 
  = {act= a, st=s, err= e}

fun extract_err e =
  let
    val (mssgs, loc_mssg) = (case e of
      SOME (err, str) => (map (fn ((_, txt), _) => YXML.content_of txt) (Runtime.exn_messages err), YXML.content_of str)
      | NONE => ([], ""));
  in implode (mssgs @ ["\n", loc_mssg, "\n"]) end;

fun get_st state = #st (hd state)
fun get_act state = #act (hd state)
fun get_err state = #err (hd state)

fun size state = List.length state

fun init thy = [make_one(
  Actions.void,
  Toplevel.make_state (SOME thy),
  "")];

fun read (txt, state) =
  let
    val st = get_st state;
    val thy = Toplevel.theory_of st;
    val acts = Actions.make thy txt
    val new_act_sts = Actions.apply_all acts st
    fun convert (act, st, e) =
      let
        val err = extract_err e
      in make_one (act, st, err) end;
    val state' = map convert new_act_sts
  in rev state' end;

fun eval (new_sts, state) = new_sts @ state;

fun print state = Get.user_state {break_lines=false} (get_st state);

fun read_eval (txt, state) = read (txt, state) 
  |> (fn state' => eval (state', state));

val repl = read_eval #> print

end;