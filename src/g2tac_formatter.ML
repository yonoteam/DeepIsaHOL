
signature G2TAC_FORMATTER =
sig
  (* Pretty printing *)
  val pretty_state: Proof.context -> Data.T -> Pretty.T
  val pretty_action: Data.T -> Pretty.T
  val pretty_step: Data.env -> Data.T -> Pretty.T
end;


structure G2tac_Formatter: G2TAC_FORMATTER =
struct

(* pretty_state: context -> Data.T -> Pretty.T *)
fun pretty_state ctxt0 data =
  let
    val state = Data.state_of data;
    val ctxt0' = if Toplevel.is_proof state 
      then Proof.context_of (Toplevel.proof_of state) 
      else ctxt0
    val ctxt = 
      ctxt0'
      |> Config.put show_sorts false
      |> Config.put show_types false 
    val goal = Json_Maker.pretty_goal ctxt data;
    val hs = Json_Maker.pretty_hyps false 0 ctxt data;
    val seen = Json_Maker.pretty_proven false 0 ctxt data;
    val cs = Json_Maker.pretty_consts false 0 ctxt data;
    val vs = Json_Maker.pretty_vars false 0 ctxt data;
    val Ts = Json_Maker.pretty_types false 0 ctxt data;
    val usr_out = Json_Maker.pretty_user_state state;
    val pretty_state = Print.pretty_json_attribs false "{" "}" 0
      [usr_out, goal, hs, seen, vs, cs, Ts]
  in Print.pretty_json_attrib false 0 "step" pretty_state end;

(* pretty_action: Data.T -> Pretty.T *)
fun pretty_action data = Pretty.enclose "{" "}" [Json_Maker.pretty_action data];

(* pretty_step: Data.env -> Data.T -> Pretty.T *)
fun pretty_step env data =
  [pretty_state (Data.ctxt_of env) data, 
    Pretty.str "OUTPUT", 
    pretty_action data, 
    Pretty.str "<END>", 
    Pretty.str "\\n"]
  |> Ops.intersp [Pretty.brk 1]
  |> Pretty.block

end;
