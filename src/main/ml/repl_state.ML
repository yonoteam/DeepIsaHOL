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
  val call_hammer: (string * string) list * T -> T * string
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

fun no_subgoals' step = Pred.no_subgoals (#st step);

fun make_one (
    n: int,
    a: Actions.T, 
    s: Toplevel.state, 
    e: string) 
  = ({num=n, act=a, st=s, err=e}:step)


(* DEFINITION *)

(* State working as a stack *)
type T = step list

exception E_Repl_State of T (* Needed for scala-isabelle *)

(* T -> _ *)
fun get_num (state:T) = #num (hd state)
fun get_st (state:T) = #st (hd state)
fun get_act (state:T) = #act (hd state)
fun get_err (state:T) = #err (hd state)


(* BASIC OPERATIONS *)

fun init thy = [make_one(
  0,
  Actions.void,
  Toplevel.make_state (SOME thy),
  "")];

fun read (txt, state) =
  let
    val n = get_num state;
    val st = get_st state;
    val thy = Toplevel.theory_of st;
    val acts = Actions.make thy txt
    val new_act_sts = Actions.timed_apply_all {timeout_in_secs=10} acts st
    fun convert (act, st, e) =
      let
        val err = Actions.extract_error e;
      in make_one (n+1, act, st, err) end;
    val state' = map convert new_act_sts
  in (rev state'):T end;

fun eval (new_sts: T, state: T) = (new_sts @ state):T;

fun str_of_one (step: step) =
  let
    val {num=n, st=s, act=a, err=e} = step
    val str_n = Value.print_int n;
    val str_a = "act=" ^ Actions.text_of a;
    val str_s = "state=" ^ Get.user_state {break_lines=false} s
    val str_e = "err=" ^ e;
    val body = Library.commas [str_n, str_a, str_s, str_e]
  in Library.enclose "<" ">" body end;

fun print (state:T) = str_of_one (hd state);

fun read_eval (txt, state) = read (txt, state) 
  |> (fn state' => eval (state', state));

val repl = read_eval #> print


(* MORE OPERATIONS *)

fun undo state = 
  let
    val n = get_num state;
    fun removable step = #num step = n andalso not (n = 0)
  in (Pred.drop_while removable state):T end;

fun reset (state:T) = Ops.last state |> Library.single;

fun go_to (thy, thy_file, act_txt) =
  let
    val empty_err_mssg = "repl.go_to: empty actions for input " ^ act_txt ^ " on location " ^ thy_file
    val thy_text = File.read (Get.os_agnostic_path thy_file);
    val acts = Actions.make_filtered thy thy_text;
    val pre_acts = Pred.take_until (Actions.on_text (Pred.contains act_txt)) acts;
    val act_st_errs = Actions.apply_all pre_acts (Toplevel.make_state (SOME thy));
    val (act, st', e) = if null act_st_errs then raise Fail empty_err_mssg
      else Ops.last act_st_errs;
    val err = Actions.extract_error e;
  in [make_one(0, act, st', err)]: T end;

fun call_hammer (params, state:T) =
  let
    val st = get_st state;
    val thy = Toplevel.theory_of st;
    val n = get_num state;
    val result = if not (Toplevel.is_proof st)
      then 
        let val mssg = "No proof state"
        in (make_one (n+1, Actions.void, st, mssg) :: state, mssg) end
      else
        let
          val proofs = Hammer.get_suggestions params st;
          fun return_if_proved ((_, prf, _) :: ps) =
            let
              val acts = Actions.make thy prf;
              val stacts = Actions.timed_apply_all {timeout_in_secs=15} acts st;
              fun convert (act, st, e) =
                let
                  val err = Actions.extract_error e;
                in make_one (n+1, act, st, err) end;
              val (state', mssg) = if Pred.starts_with "by" prf 
                then 
                  if Actions.by_finishes (hd acts) st 
                  then ((rev (map convert stacts)) @ state, "Used: '" ^ prf ^ "'")
                  else return_if_proved ps
                else if Actions.no_errors stacts 
                  then ((rev (map convert stacts)) @ state, "Used: '" ^ prf ^ "'")
                else return_if_proved ps     
            in (state', mssg) end
          | return_if_proved [] =
            let 
              val mssg = "No valid hammer proof"
              val state' = make_one (n+1, Actions.void, st, mssg) :: state;
            in (state', mssg) end
        in return_if_proved proofs end;
  in result end;


(* INFORMATION RETRIEVAL *)

fun size (state:T) = List.length state

fun is_at_proof (state:T) = is_at_proof' (hd state)
fun no_subgoals (state:T) = no_subgoals' (hd state)

fun last_action state = Actions.text_of (get_act state)
fun last_state state = Get.user_state {break_lines=false} (get_st state)
fun last_error state = get_err state

fun to_string (state:T) = 
  let
    val new_line = "\n"
    fun do_one step = new_line ^ str_of_one step
    val pre_final = map do_one state
  in Library.commas pre_final end;

fun pretty (state:T) = Pretty.str (to_string state);

(* (* T needs to be a simple type (e.g. datatype) in order to do this *)
val _ =  ML_system_pp (fn _ => fn _ => Pretty.to_polyml o pretty) *)

fun proof_so_far (state:T) =
  let
    val err_message = "Repl_State.proof_so_far error: not in proof state."
    val result = if not (is_at_proof' (hd state)) 
      then err_message
      else let
        val acts = (Pred.take_while is_at_proof' state) 
        |> rev |> map (Actions.text_of o #act)
      in implode (Ops.intersp ["\n"] acts) end;
  in result end;

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