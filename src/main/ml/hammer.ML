(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Operations involving Sledgehammer
*)

signature HAMMER =
sig
  val extract_proof: string -> string * string * int
  val remove_time_of: string -> string
  val get_suggestions: (string * string) list -> Toplevel.state -> (string * string * int) list
  val get_useful_facts: int -> Toplevel.state -> (string * thm) list
end

structure Hammer: HAMMER =
struct

fun contains_time input_str = 
  Pred.contains " ms)" input_str 
  orelse Pred.contains " s)" input_str

fun remove_time_of s =
  if not (contains_time s) then s
  else
    let
      val ss = Substring.full s
      val (prefix, _) = Substring.splitr (fn c => c <> #"(") ss
    in if Substring.isEmpty prefix then s 
      else Substring.string (Substring.trimr 1 prefix)
    end;

fun extract_proof s = 
  if not (contains_time s) then raise Fail ("No proof to extract from: " ^ s)
  else let
    val ss = Substring.full s
    val (prover, rest) = Substring.splitl (fn c => c <> #":") ss;
    val (proof_ss, time_ss) = Substring.splitr (fn c => c <> #"(") rest;
    val proof = Substring.string proof_ss
      |> Ops.drop_untilL' "Try this:"
      |> Print.yxml_to_string
      |> Ops.stripLR " " "(, ";
    val time =
      let
        val time_str = Substring.string time_ss;
        val t = Ops.stripLR " " " )" time_str
        val is_milliseconds = Pred.contains "ms" t
        val num_opt = Real.fromString t
      in case num_opt of
        NONE => 0.0
        | SOME n => if is_milliseconds 
          then n * 1000.0 else n * 1000000.0
      end
  in (Substring.string prover, proof, Real.ceil time) end;

fun silence_methods debug =
  Config.put Metis_Tactic.verbose debug
  #> not debug ? (fn ctxt =>
      ctxt
      |> Simplifier_Trace.disable
      |> Context_Position.set_visible false
      |> Config.put Unify.unify_trace false
      |> Config.put Argo_Tactic.trace "none")

(* from try0 Isabelle2025 *)
val silence_state =
  Proof.map_contexts (silence_methods false #> Config.put SMT_Config.verbose false);

fun get_suggestions params st =
  let
    val prover_outputs = Synchronized.var "suggestions" ([] : string list);
    fun register msg = Synchronized.change prover_outputs (fn msgs => msg :: msgs);
    val thy = Toplevel.theory_of st;
    val params' = Sledgehammer_Commands.default_params thy params;
    val _ = Sledgehammer.run_sledgehammer 
      params' 
      Sledgehammer_Prover.Normal 
      (SOME register) 
      1
      Sledgehammer_Fact.no_fact_override
      (silence_state (Toplevel.proof_of st));
    val suggestions = filter contains_time (Synchronized.value prover_outputs);
    val sort_on_time = Library.sort (fn ((_, _, i),(_, _, j)) => Library.int_ord (i, j));
  in sort_on_time (map extract_proof suggestions) end;

fun get_useful_facts n st =
  let
    val ctxt = Toplevel.context_of st;
    val thy = Toplevel.theory_of st;
    val goal = #goal (Proof.goal (Toplevel.proof_of st));
    val (_, hyp_ts, concl_t) = ATP_Util.strip_subgoal goal 1 ctxt;
    val params = Sledgehammer_Commands.default_params thy [];
    val override = Sledgehammer_Fact.no_fact_override;
    val all_thms = Sledgehammer_Fact.nearly_all_facts_of_context
      ctxt false override [] hyp_ts concl_t;
    val pre_facts = Sledgehammer_MaSh.relevant_facts 
      ctxt params "" n override hyp_ts concl_t all_thms;
    val facts = pre_facts
      |> maps (fn (_, hammer_fact_list) => hammer_fact_list)
      |> map (fn ((name, _), thm) => (name, thm))
      |> Library.distinct (fn ((name1, _), (name2, _)) => name1 = name2)
      |> Library.take n;
  in facts end;

end