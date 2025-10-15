(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Operations interacting with an LLM
*)

signature LARGE_LANG_MODEL =
sig
  val json_text: Actions.stact list -> string
  val json_text_at: theory -> string -> int -> string
  val prove: int -> Actions.stact list -> string
end;


structure LLM: LARGE_LANG_MODEL =
struct

fun json_text stacts =
  if null stacts then raise Fail "llm.json_text received null input"
  else let
    val (_, st, _) = Ops.last stacts;
    val prf_so_far_att = stacts
      |> map (fn (a,_,_): Actions.stact => Actions.text_of a)
      |> Library.space_implode " "
      |> Print.make_parseable
      |> Pretty.str |> Print.pretty_json_elem "proof_so_far";
    val usr_st_att = 
      Get.user_state {break_lines=false} st
      |> Print.make_parseable
      |> Pretty.str |> Print.pretty_json_elem "user_state";
    val prf_data =
      Print.pretty_json_attribs false "{" "}" 0 [usr_st_att, prf_so_far_att]
      |> Print.string_of_pretty;
  in prf_data end;

fun json_text_at thy thy_file line_num =
  Actions.proof_at thy thy_file line_num
  |> json_text;

(* decides whether the by-suggestions are (partially) correct *)
fun do_next_from_llm act' (finished, last_st, resp_so_far) =
  if finished then (finished, last_st, resp_so_far)
  else if Actions.on_kind (Pred.is "by") act'
  then let 
    val thy = Toplevel.theory_of last_st;
    val ctxt = Toplevel.context_of last_st;
    val new_acts = Actions.make thy (Actions.by_to_apply ctxt act');
    val _ = Output.tracing ("Turned into '" ^ implode (map Actions.text_of new_acts) ^ "'")
    val st_acts = Actions.timed_apply_all {timeout_in_secs=10} new_acts last_st;
    fun build_next (a', s', es') (proved, so_far) =
      if proved then (proved, so_far)
      else if Pred.no_subgoals s' then (true, so_far @ [(a', s', es')])
      else if null es' then (false, so_far @ [(a', s', es')])
      else (false, so_far)
    val (proved, pre_stacts) = fold build_next st_acts (false, []);
    fun recover stacts = 
      Actions.make_one thy ("by" ^ (unprefix "apply" (Actions.text_of (#1 (hd stacts)))))
      |> (fn act => Actions.apply_all [act] last_st);
    fun get_last_st stacts = #2 (Ops.last stacts);
    val result = if null pre_stacts 
        then (finished, last_st, resp_so_far)
      else if length pre_stacts < length st_acts
        then if not proved 
          then (finished, get_last_st pre_stacts, resp_so_far @ pre_stacts)
          else (true, get_last_st pre_stacts, resp_so_far @ (recover pre_stacts))
      else if not proved 
        then (finished, get_last_st pre_stacts, resp_so_far @ pre_stacts)
      else (true, get_last_st pre_stacts, resp_so_far @ (Actions.apply_all [act'] last_st))
    in result end
  else let
      val (_, st', errs') = hd (Actions.timed_apply_all {timeout_in_secs=10} [act'] last_st);
    in if null errs' 
      then (Pred.no_subgoals st', st', resp_so_far @ [(act', st', errs')])
      else (finished, last_st, resp_so_far)
    end;

fun llm_dfs max_depth proved (depth, prf_so_far) =
  if proved then prf_so_far
  else if depth > max_depth then prf_so_far @ Actions.apply_oops (#2 (Ops.last prf_so_far))
  else let
    val prf_data = json_text prf_so_far;
    val last_st = #2 (Ops.last prf_so_far);
    val _ = Client.communicate prf_data;
    val llm_response = Client.get_last_response ();
    val result = (case llm_response of
      SOME response => 
        let
          val thy = Toplevel.theory_of last_st;
          val depth_report = "at depth " ^ Value.print_int depth;
          val _ = Output.tracing ("Trying suggestion '" ^ response ^ "' " ^ depth_report)
          val response_acts =
            Actions.make thy response
            |> filter (not o Actions.is_malformed)
          val (proved', _, final_stacts) = 
            fold do_next_from_llm response_acts (false, last_st, []);
        in llm_dfs max_depth proved' (depth + 1, prf_so_far @ final_stacts) end
      | NONE => prf_so_far)
  in result end;

fun prove max_depth prf_so_far =
  let
    val prf_so_far' = llm_dfs max_depth false (1, prf_so_far);
    val to_concat = prf_so_far'
      |> filter (fn (a, _, _) => Actions.on_kind (Pred.is_not "<ignored>") a)
      |> map (fn (a, _, _) => Actions.text_of a ^ "\n  ");
    val pref = implode (map (fn (a, _, _) => Actions.text_of a) prf_so_far);
  in Library.unprefix (pref ^ "\n") (implode to_concat) end;


end;