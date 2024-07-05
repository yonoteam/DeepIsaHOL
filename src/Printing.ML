
signature PRINTING =
sig
  datatype mode = FILE of string | OUT;
  datatype format = YXML of int | TRM | AST | GOAL;
  val term_string: Proof.context -> format -> term -> string
  val thm_string: Proof.context -> format -> thm -> string
  val print_str: mode -> string -> unit
end;


structure Printing: PRINTING =
struct

datatype mode = FILE of string | OUT;

datatype format = YXML of int | TRM | AST | GOAL;

exception FORMAT of string

val yx_string = Syntax.string_of_term

(* Proof.context -> format -> trm -> string *)
fun term_string ctxt format t =
  let 
    val result = (case format of
        YXML depth => Pretty.string_of (XML.pretty depth (YXML.parse (yx_string ctxt t)))
      | TRM => (YXML.content_of (yx_string ctxt t))
      | AST => (ML_Syntax.print_term t)
      | GOAL => raise FORMAT("Wrong format GOAL for type term.")
    )
  in result
  end;

(* Proof.context -> format -> thm -> string *)
fun thm_string ctxt format thm =
  let 
    val main_term = Thm.prop_of thm;
    val result = (case format of
        GOAL => YXML.content_of (Pretty.string_of (Goal_Display.pretty_goal ctxt thm))
      | _ => term_string ctxt format main_term
    )
  in result
  end;

(* mode -> string -> unit *)
fun print_str mode str =
  let 
    val _ = (case mode of
        OUT => writeln str
      | FILE name => let 
          val file = TextIO.openAppend(name) 
        in TextIO.output(file, str ^ "\n"); TextIO.closeOut(file)
        end)
  in () 
  end;

(* Toplevel.state -> {consts: Pretty.T list, goal: Pretty.T, hyps: Pretty.T list, tvars: Pretty.T list, vars: Pretty.T list} *)
fun get_goal_data state = 
  let
    val proof_state = Toplevel.proof_of state;
    val ctxt = ((Proof.context_of proof_state) 
      |> Config.put show_sorts false 
      |> Config.put show_types false);
    val prop = Proof.goal (Toplevel.proof_of state);
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val goal_term = hd goal_hyps;
    val (Phis, Phi) = Logic.strip_horn goal_term;
    val xs = Data_Ops.get_vars goal_hyps;
    val cs = Data_Ops.get_consts goal_hyps;
    val Ts = Data_Ops.get_atomic_types goal_hyps;
  in {goal = Syntax.pretty_term ctxt goal_term, 
   consts = map (Syntax.pretty_term ctxt) cs, 
   vars = map (Syntax.pretty_term (Config.put show_types true ctxt)) xs, 
   tvars = map (Syntax.pretty_typ ctxt) Ts, 
   hyps = map (Syntax.pretty_term ctxt) (tl goal_hyps)} end;

end;