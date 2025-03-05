(* Mantainers:
 *   Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
 *
 * Read-eval-print-loop state (to be called by scala-isabelle)
 *)

signature REPL_STATE =
sig
  type T
  exception E_Repl_State of T
  val get_st: T -> Toplevel.state
  val get_act: T -> Actions.T
  val get_err: T -> string
  val init: theory -> T
  val size: T -> int
  val read: string * T -> T
  val eval: T * T -> T
  val print: T -> string
  val read_eval: string * T -> T
  val repl: string * T -> string
  val undo: T -> T
  val reset: T -> T
  val str_of: T -> string
  val pretty: T -> Pretty.T
end;


structure Repl_State: REPL_STATE =
struct

(* TODO: improve *)
(* State working as a stack *)

type step = {
    num: int,
    act: Actions.T, 
    st: Toplevel.state, 
    err: string
  }

type T = step list

(* Needed for scala-isabelle *)
exception E_Repl_State of T

fun make_one (
    n: int,
    a: Actions.T, 
    s: Toplevel.state, 
    e: string) 
  = {num=n, act=a, st=s, err=e}

fun extract_err e =
  let
    fun add_loc txts txt = if txt = "" then txts else txts @ ["\n", txt, "\n"]
    val (mssgs, loc_mssg) = (case e of
      SOME (err, str) => (map (fn ((_, txt), _) => YXML.content_of txt) (Runtime.exn_messages err), YXML.content_of str)
      | NONE => ([], ""));
  in implode (add_loc mssgs loc_mssg) end;

fun get_num state = #num (hd state)
fun get_st state = #st (hd state)
fun get_act state = #act (hd state)
fun get_err state = #err (hd state)

(* size: T -> int *)
fun size state = List.length state

(* init: theory -> T *)
fun init thy = [make_one(
  0,
  Actions.void,
  Toplevel.make_state (SOME thy),
  "")];

(* read: string * T -> T *)
fun read (txt, state) =
  let
    val n = get_num state;
    val st = get_st state;
    val thy = Toplevel.theory_of st;
    val acts = Actions.make thy txt
    val new_act_sts = Actions.apply_all acts st
    fun convert (act, st, e) =
      let
        val err = extract_err e
      in make_one (n+1, act, st, err) end;
    val state' = map convert new_act_sts
  in rev state' end;

(* eval: T * T -> T *)
fun eval (new_sts, state) = new_sts @ state;

(* print: T -> string *)
fun print state = Get.user_state {break_lines=false} (get_st state);

(* read_eval: string * T -> T *)
fun read_eval (txt, state) = read (txt, state) 
  |> (fn state' => eval (state', state));

(* repl: string * T -> string *)
val repl = read_eval #> print

(* undo: T -> T *)
fun undo state = 
  let
    val n = get_num state;
    fun removable step = #num step = n andalso not (n = 0)
  in Library.drop_prefix removable state end;

(* reset: T -> T *)
fun reset state = Ops.last state |> Library.single

fun str_of (state:T) = 
  let
    fun str_of_one step =
      let
        val {num=n, st=_, act=a, err=e} = step
        val str_n = Value.print_int n;
        val str_a = "act=" ^ Actions.text_of a;
        val str_e = "err=" ^ e;
        val body = Library.commas [str_n, str_a, str_e]
      in Library.enclose "\n<" ">" body end;
    val pre_final = map str_of_one state
  in Library.commas pre_final end;

fun pretty (state:T) = Pretty.str (str_of state)

(* (* T needs to be a simple type (e.g. datatype) in order to do this *)
val _ =  ML_system_pp (fn _ => fn _ => Pretty.to_polyml o pretty)
 *)
end;