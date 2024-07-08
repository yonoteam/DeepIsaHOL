
signature PRINTING =
sig
  datatype mode = FILE of string | OUT;
  datatype format = YXML of int | TRM | AST | GOAL;
  val term_string: Proof.context -> format -> term -> string
  val thm_string: Proof.context -> format -> thm -> string
  val pretty_state: Proof.context -> Toplevel.state -> Pretty.T
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

local

(* Proof.context -> {show_sorts: bool, show_types: bool} 
  -> (Proof.context -> 'key -> Pretty.T) -> (Proof.context -> 'val -> Pretty.T) 
  -> ('key * 'val list) list -> Pretty.T list *)
fun pretty_alist ctxt config pretty_key pretty_val alist = 
  let
    val {show_types = showT, show_sorts = showS} = config;
    val ctxt' = ctxt
      |> Config.put show_sorts showS
      |> Config.put show_types showT
    fun pretty_entry (k, vs) = Pretty.block 
      (Pretty.commas (map (pretty_val ctxt') vs) 
      @ [Pretty.str " ::", Pretty.brk 1, (pretty_key ctxt') k]);
  in map pretty_entry alist end;

fun const_info t = (case t of Const (_, T) => (t, T) | _ => raise TERM("not a constant", [t]));

(* term -> term * typ *)
fun vars_info t = (case t of Var (_, T) => (t,T)  | Free (_, T) => (t, T) | _ => raise TERM("not a var", [t]));


(* Proof.context -> term list -> Pretty.T *)
fun pretty_consts ctxt consts = pretty_alist ctxt {show_types = true, show_sorts = false} Syntax.pretty_typ Syntax.pretty_term (Data_Ops.flatten const_info consts);

fun pretty_vars ctxt vars = pretty_alist ctxt {show_types = false, show_sorts = false} Syntax.pretty_typ Syntax.pretty_term (Data_Ops.flatten vars_info vars);


in

(* get_goal_data = fn: Toplevel.state -> {consts: term list, goal: term, hyps: term list, tvars: typ list, vars: term list} *)
fun get_goal_data state = 
  let
    val prop = Proof.goal (Toplevel.proof_of state);
    val goal_hyps = map Thm.prop_of (#goal prop :: #facts prop);
    val xs = Data_Ops.get_vars goal_hyps;
    val cs = Data_Ops.get_consts goal_hyps;
    val Ts = Data_Ops.get_atomic_types goal_hyps;
  in {goal = hd goal_hyps, consts = cs, vars = xs, tvars = Ts, hyps = tl goal_hyps} end;

fun pretty_state ctxt state = 
  let
    val state_data = get_goal_data state;
    fun jpair name prt_attribs = Pretty.block [Pretty.quote (Pretty.str name), Pretty.str ":", Pretty.brk 1, prt_attribs];
    val consts = jpair "constants" (Pretty.enum "," "{" "}" (pretty_consts ctxt (#consts state_data)));
    val vars = jpair "variables" (Pretty.enum "," "{" "}" (pretty_vars ctxt (#vars state_data)));
    val goal = jpair "term" (Syntax.pretty_term ctxt (#goal state_data));
  in jpair "state" (Pretty.enum "," "{" "}" [goal, vars, consts]) end;

end;


end;