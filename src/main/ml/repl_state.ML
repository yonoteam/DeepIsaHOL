(* Mantainers:
 *   Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
 *
 * Read-eval-print-loop state (to be called by scala-isabelle)
 *)

signature REPL_STATE =
sig
  type T
  exception E_Repl_State of T
  val get_num: T -> int
  val get_st: T -> Toplevel.state
  val get_act: T -> Actions.T
  val get_err: T -> string
  val is_at_proof: T -> bool
  val no_subgoals: T -> bool
  val last_action: T -> string
  val last_state: T -> string
  val last_error: T -> string
  val size: T -> int

  val init: theory -> T
  val read: string * T -> T
  val eval: T * T -> T
  val print: T -> string
  val read_eval: string * T -> T
  val repl: string * T -> string
  val undo: T -> T
  val reset: T -> T
  val go_to: (theory * string * string) -> T
  val to_string: T -> string
  val pretty: T -> Pretty.T
  val proof_so_far: T -> string
  val last_proof_of: T -> string
end;


structure Repl_State: REPL_STATE =
struct


(* REPL STEP *)

(* TODO: improve *)
type step = {
    num: int,
    act: Actions.T, 
    st: Toplevel.state, 
    err: string
  }

fun is_at_proof' (step:step) = Toplevel.is_proof (#st step);

fun no_subgoals' step = 
  let
    val thm = #goal (Proof.goal (Toplevel.proof_of (#st step)))
    val prop = Thm.prop_of thm;
    val (As, _) = Logic.strip_horn prop;
  in is_at_proof' step andalso  null As end;

fun make_one (
    n: int,
    a: Actions.T, 
    s: Toplevel.state, 
    e: string) 
  = ({num=n, act=a, st=s, err=e}:step)

fun extract_err e =
  let
    fun add_loc txts txt = 
      if txt = "" 
      then txts 
      else txts @ ["\n", txt, "\n"]
    fun message_of err = 
      Runtime.exn_messages err
      |> map (fn ((_, txt), _) => Get.string_of_yxml txt)
    val (mssgs, loc_mssg) = (case e of
      SOME (err, str) => (message_of err, Get.string_of_yxml str)
      | NONE => ([], ""));
  in implode (add_loc mssgs loc_mssg) end;


(* DEFINITION *)

(* State working as a stack *)
type T = step list

exception E_Repl_State of T (* Needed for scala-isabelle *)

(* T -> _ *)
fun get_num state = #num (hd state)
fun get_st state = #st (hd state)
fun get_act state = #act (hd state)
fun get_err state = #err (hd state)


(* BASIC OPERATIONS *)

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
    val new_act_sts = Actions.timed_apply_all {timeout_in_secs=10} acts st
    fun convert (act, st, e) =
      let
        val err = extract_err e
      in make_one (n+1, act, st, err) end;
    val state' = map convert new_act_sts
  in rev state' end;

(* eval: T * T -> T *)
fun eval (new_sts, state) = new_sts @ state;

(* str_of_one: step -> string *)
fun str_of_one (step: step) =
  let
    val {num=n, st=s, act=a, err=e} = step
    val str_n = Value.print_int n;
    val str_a = "act=" ^ Actions.text_of a;
    val str_s = "state=" ^ Get.user_state {break_lines=false} s
    val str_e = "err=" ^ e;
    val body = Library.commas [str_n, str_a, str_s, str_e]
  in Library.enclose "<" ">" body end;

(* print: T -> string *)
fun print state = str_of_one (hd state);

(* read_eval: string * T -> T *)
fun read_eval (txt, state) = read (txt, state) 
  |> (fn state' => eval (state', state));

(* repl: string * T -> string *)
val repl = read_eval #> print


(* MORE OPERATIONS *)

(* undo: T -> T *)
fun undo state = 
  let
    val n = get_num state;
    fun removable step = #num step = n andalso not (n = 0)
  in Pred.drop_while removable state end;

(* reset: T -> T *)
fun reset state = Ops.last state |> Library.single;

(* go_to: (theory * string * string) -> T *)
fun go_to (thy, thy_path, act_txt) =
  let
    val empty_err_mssg = "repl.go_to: empty actions for input " ^ act_txt ^ " on location " ^ thy_path
    val thy_text = File.read (Path.explode thy_path);
    val acts = Actions.make' thy thy_text;
    val pre_acts = Pred.take_until (Actions.on_text (Pred.contains act_txt)) acts;
    val act_st_errs = Actions.apply_all pre_acts (Toplevel.make_state (SOME thy));
    val (act, st, e) = if null act_st_errs then raise Fail empty_err_mssg
      else Ops.last act_st_errs;
    val err = extract_err e;
  in [make_one(0, act, st, err)] end;


(* INFORMATION RETRIEVAL *)

(* size: T -> int *)
fun size state = List.length state

(* T -> bool *)
fun is_at_proof state = is_at_proof' (hd state)
fun no_subgoals state = no_subgoals' (hd state)

(* T -> string *)
fun last_action state = Actions.text_of (get_act state)
fun last_state state = Get.user_state {break_lines=false} (get_st state)
fun last_error state = get_err state

(* to_string: T -> string *)
fun to_string (state:T) = 
  let
    val new_line = "\n"
    fun do_one step = new_line ^ str_of_one step
    val pre_final = map do_one state
  in Library.commas pre_final end;

(* pretty: T -> Pretty.T *)
fun pretty (state:T) = Pretty.str (to_string state);

(* (* T needs to be a simple type (e.g. datatype) in order to do this *)
val _ =  ML_system_pp (fn _ => fn _ => Pretty.to_polyml o pretty) *)

(* proof_so_far: T -> string *)
fun proof_so_far state =
  let
    val err_message = "Repl_State.proof_so_far error: not in proof state."
    val result = if not (is_at_proof' (hd state)) 
      then err_message
      else let
        val acts = (Pred.take_while is_at_proof' state) 
        |> rev |> map (Actions.text_of o #act)
      in implode (Ops.intersp ["\n"] acts) end;
  in result end;

(* last_proof_of: T -> string *)
fun last_proof_of (state: T) =
  let
    fun find_proof (step :: step' :: steps) =
      if is_at_proof' step then ([], step :: step' :: steps)
      else if is_at_proof' step' then ([step], step' :: steps)
      else find_proof (step' :: steps)
      | find_proof _ = ([], [])
    val (start_ls, state') = find_proof state;
    val result = case state' of
      [] => ""
      | step :: steps => 
        let
          val pre_prf = proof_so_far (step :: steps)
          val final = if null start_ls then "" 
            else "\n" ^ (Actions.text_of (#act (hd start_ls)))
        in pre_prf ^ final end;
  in result end;

end;