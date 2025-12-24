(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Representation of Isabelle's transitions as user actions.
*)

signature ACTIONS =
sig
  (* Actions.T *)
  type T
  val void: T
  val text_of: T -> string
  val transition_of: T -> Toplevel.transition
  val kind_of: T -> string
  val pos_of: T -> Position.T
  val on_text: (string -> 'a) -> T -> 'a
  val on_kind: (string -> 'a) -> T -> 'a
  val by_to_apply: Proof.context -> T -> string
  val does_something: theory -> T -> bool
  val is_malformed: T -> bool
  val is_ml: T -> bool
  val make: theory -> string -> T list
  val make_one: theory -> string -> T
  val make_filtered: theory -> string -> T list
  val make_upto: theory -> string -> int -> T list

  (* Application *)
  type stact = T * Toplevel.state * Runtime.error list
  val apply: T -> Toplevel.state -> Toplevel.state * Runtime.error list
  val apply': T -> Toplevel.state -> Toplevel.state
  val apply_all: T list -> Toplevel.state -> stact list
  val apply_while: (stact -> bool) -> T list -> Toplevel.state -> stact list * T list
  val timed_apply_all: {timeout_in_secs: int} -> T list -> Toplevel.state -> stact list
  val apply_oops: Toplevel.state -> stact list
  val report_acts: stact list -> string list
  val extract_error: Runtime.error list -> string
  val report_errors: stact list -> string list
  val no_errors: stact list -> bool
  val proof_at: theory -> string -> int -> stact list
  val by_finishes: T -> Toplevel.state -> bool
end;


structure Actions: ACTIONS =
struct


(** ACTIONS **)

datatype T = Action of {
  text: string,
  transition: Toplevel.transition}

fun build txt tr = Action {text=txt, transition=tr};

val void = build "" (Toplevel.ignored Position.none);

fun text_of (Action a) = #text a;

fun transition_of (Action a) = #transition a;

fun kind_of (Action a) = Toplevel.name_of (#transition a);

fun pos_of (Action a) = Toplevel.pos_of (#transition a);

fun on_text f act = f (text_of act);

fun on_kind f act = f (kind_of act);

fun by_to_apply ctxt act =  
  let
    val mk_toks = Get.tokens {filtered=true} ctxt;
    val untoks = Print.coalesce_method_txt o map Token.unparse;
    val no_by_txt = unprefix "by" (text_of act);
    (* adding an `apply` at the end to make the parsers work correctly *)
    val toks_to_parse = mk_toks (no_by_txt ^ " apply"); 
    val untoked_txt = untoks toks_to_parse;
    val opt_parsed = try (Method.parse -- Scan.option Method.parse) toks_to_parse;
    val result = (case opt_parsed of 
      SOME ((_, opt_m2), rest_toks) => (case opt_m2 of 
        SOME _ => let
            val m1_rest_txt = untoks (snd (Method.parse toks_to_parse));
            val m2_txt = unsuffix (untoks rest_toks) m1_rest_txt;
            val m1_txt = unsuffix m1_rest_txt untoked_txt;
          in "apply " ^ m1_txt ^ "\n  apply " ^ m2_txt end
        | NONE => "apply " ^ (unsuffix (untoks rest_toks) untoked_txt))
      | NONE => "apply" ^ no_by_txt)
  in result end;

fun make thy0 thy_text =
  let
    val trs = Get.transitions thy0 Position.start thy_text;
    fun dist tr1 tr2 = 
      Position.distance_of (Toplevel.pos_of tr1, Toplevel.pos_of tr2)
      |> Option.valOf;
    fun record_acts syms [tr] = [build (implode syms) tr]
      | record_acts _ [] = []
      | record_acts syms (tr1 :: tr2 :: trs) = 
          let 
           val (syms1, syms2) = Library.chop (dist tr1 tr2) syms;
           val act = build (implode syms1) tr1;
          in act :: record_acts syms2 (tr2 :: trs) end;
  in record_acts (Symbol.explode thy_text) trs end;

fun make_one thy0 thy_text = hd (make thy0 thy_text)

fun does_something thy0 = on_kind (
  Pred.is_not "<ignored>" 
  andf Pred.neg (Seps.is_document_step (Get.grouped_commands thy0)));

val is_malformed = on_kind (Pred.is "<malformed>");

val is_ml = on_kind (Pred.is "ML" orf Pred.is "ML_val")

fun make_filtered thy0 thy_text = void :: (filter (does_something thy0) (make thy0 thy_text));

fun make_upto thy0 thy_file line_num =
  if line_num <= 0 then raise Fail ("Invalid line number " ^ (Value.print_int line_num))
  else let
    fun is_below n a = n < the (Position.line_of (Toplevel.pos_of (transition_of a)))
    val acts = 
      make thy0 (File.read (Get.os_agnostic_path thy_file))
      |> Pred.take_until' (is_below line_num)
  in acts end;


(** APPLICATION **)

type stact = T * Toplevel.state * Runtime.error list (* stateful actions *)

fun apply (Action a) st = 
  let 
    val (errs', opt_st) = Toplevel.command_errors true (#transition a) st;
    val final_st = (case opt_st of
      NONE => st
      | SOME st' => st')
  in (final_st, errs') end;

fun apply' act st = fst (apply act st);

fun iter_next apply_fun act' (_: T, st: Toplevel.state, _: Runtime.error list) =
  let
    val (st', errs') = apply_fun act' st;
  in (act', st', errs') end;

fun apply_all [] _ = []
  | apply_all acts opt_st = Ops.log_fold (iter_next apply) (void, opt_st, []) acts;

fun apply_while P acts opt_st = Pred.log_fold P (iter_next apply) acts (void, opt_st, []);

fun timed_apply_all {timeout_in_secs=t} acts st =
  let
    val t' = Time.fromSeconds t
    fun timeout_apply act' st = case Ops.apply_with_timeout t' (apply act') st of
        Exn.Exn exn => (st, Runtime.exn_messages exn)
      | Exn.Res result => result;
  in Ops.log_fold (iter_next timeout_apply) (void, st, []) acts end;

fun apply_oops st = apply_all (make (Toplevel.theory_of st) "oops") st;

fun report_acts (stacts: stact list) = map (fn (a, _, _) => text_of a) stacts

fun make_error_str ((num, err), str_opt) =
  let
    val str1 = Value.print_int num ^ ": "
    val str2 = (Print.yxml_to_string err)
  in (case str_opt of
    SOME txt => str1 ^ str2 ^ ", opt: " ^ (Print.yxml_to_string txt)
    | NONE => str1 ^ str2) end;

fun extract_error (mssgs:Runtime.error list) = String.concatWith " | " (map make_error_str mssgs);

fun report_errors (stacts: stact list) = map (fn (_, _, e) => extract_error e) stacts;

fun no_errors (stacts: stact list) = Pred.is "" (Library.space_implode "" (report_errors stacts));

fun proof_at thy thy_file line_num =
  make_upto thy thy_file line_num
  |> filter_out (is_ml orf Pred.neg (does_something thy))
  |> (fn acts => apply_all acts (Toplevel.make_state NONE))
  |> rev
  |> Pred.take_while (fn (_, st, _) => Toplevel.is_proof st)
  |> rev;

fun by_finishes act st =
  if not (on_kind (Pred.is "by") act)
  then raise Fail ("Actions.by_finishes: received non-by action: " ^ text_of act)
  else 
    let
      val thy = Toplevel.theory_of st;
      val ctxt = Toplevel.context_of st;
      val new_acts = make thy (by_to_apply ctxt act);
      val stacts = timed_apply_all {timeout_in_secs=15} new_acts st;
    in no_errors stacts andalso Pred.no_subgoals (#2 (Ops.last stacts)) end;

end;