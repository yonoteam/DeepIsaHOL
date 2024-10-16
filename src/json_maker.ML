
signature JSON_MAKER =
sig
  (* Pretty printing *)
  val pretty_action: Data.T -> Pretty.T
  val pretty_user_state: Toplevel.state -> Pretty.T
  val pretty_goal: Proof.context -> Data.T -> Pretty.T
  val pretty_hyps: bool -> int -> Proof.context -> Data.T -> Pretty.T
  val pretty_proven: bool -> int -> Proof.context -> Data.T -> Pretty.T
  val pretty_consts: bool -> int -> Proof.context -> Data.T -> Pretty.T
  val pretty_vars: bool -> int -> Proof.context -> Data.T -> Pretty.T
  val pretty_types: bool -> int -> Proof.context -> Data.T -> Pretty.T
  val pretty_deps: bool -> int -> Proof.context -> Data.env -> Pretty.T
  val pretty_methods: bool -> int -> Data.env -> Pretty.T
  val pretty_apply_kwrds: bool -> int -> Data.env -> Pretty.T
  val pretty_isar_kwrds: bool -> int -> Data.env -> Pretty.T
  val pretty_step: bool -> int -> Proof.context -> Data.T -> Pretty.T
  val pretty_proof: bool -> int -> Data.env -> Data.T Sections.T -> Pretty.T
  val json_proof: (Data.env * Data.T Sections.T) -> string
end;


structure Json_Maker: JSON_MAKER =
struct


(** PRETTY PRINTING **)

local

fun not_member_cons_if P x xs = Ops.insert_if (P andf (Pred.neg (member (op =) xs))) x xs;

(* alist_by: ('a -> ('val, 'key)) -> 'a list -> ('key * 'val list) list *)
fun alist_by extract_info xs =
let
  fun register x [] = 
      let
        val (name, key) = extract_info x;
      in [(key, [name])] end
    | register x ((key', names) :: kns) =
      let
        val (name, key) = extract_info x;
      in if key' = key 
        then (key', not_member_cons_if (fn _ => true) name names) :: kns  
        else (key', names) :: register x kns end;
in fold register xs [] end;

(* const_info: term -> term * typ *)
fun const_info t = (case t of 
  Const (_, T) => (t, T) 
  | _ => raise TERM("not a constant", [t]));

(* vars_info: term -> term * typ *)
fun vars_info t = (case t of 
  Var (_, T) => (t,T)  
  | Free (_, T) => (t, T) 
  | _ => raise TERM("not a var", [t]));

(* types_info: typ -> typ * sort *)
fun types_info T = (case T of 
  TVar (_, S) => (T, S) 
  | TFree (_, S) => (T, S) 
  | _ => raise TYPE("not a type variable", [T], []));

(* pretty_alist: Proof.context 
  -> (Proof.context -> 'key -> Pretty.T) 
  -> (Proof.context -> 'val -> Pretty.T) 
  -> string -> ('key * 'val list) list -> Pretty.T list *)
fun pretty_alist ctxt pretty_key pretty_val str alist = 
  let
    fun pretty_right k vs = Pretty.block (
      Pretty.str "\"" :: Pretty.commas (map (pretty_val ctxt) vs) 
      @ [Pretty.brk 1, Pretty.str "::", Pretty.brk 1, (pretty_key ctxt) k, Pretty.str "\""]
    );
    fun pretty_entry (n, (k, vs)) 
      = Print.pretty_json_elem (str^(Value.print_int n)) (pretty_right k vs)
  in map pretty_entry (Ops.enumerate alist) end;

(* pretty_consts_pre: Proof.context -> term list -> Pretty.T *)
fun pretty_consts_pre ctxt consts = pretty_alist ctxt
  Syntax.pretty_typ Syntax.pretty_term "Type" (alist_by const_info consts);

(* pretty_vars_pre: Proof.context -> term list -> Pretty.T *)
fun pretty_vars_pre ctxt vars = pretty_alist ctxt
  Syntax.pretty_typ Syntax.pretty_term "Type" (alist_by vars_info vars);

(* pretty_types_pre: Proof.context -> typ list -> Pretty.T list *)
fun pretty_types_pre ctxt typs = pretty_alist ctxt
  Syntax.pretty_sort Syntax.pretty_typ "Sort" (alist_by types_info typs);

(* pretty_keywords: unit -> string list -> Pretty.T list *)
val pretty_keywords = (map (Print.pretty_json_elem "name" o Pretty.quote o Pretty.str));

(* pretty_terms: Proof.context -> string list -> Pretty.T list *)
fun pretty_terms ctxt 
  = (map (Print.pretty_json_elem "term" o Pretty.quote o Pretty.unbreakable o Syntax.pretty_term ctxt));

(* pretty_fact: Proof.context -> (string * int) * thm -> Pretty.T list *)
fun pretty_fact ctxt ((name, i),thm) = [
  Print.pretty_json_elem "name" (
    Pretty.quote (Pretty.block [
      Pretty.marks_str (#1 (Proof_Context.markup_extern_fact ctxt name), name),
      Pretty.str (Thm_Name.print (name, i))])
  ),
  Print.pretty_json_elem "term" (Pretty.quote ((Pretty.unbreakable o (Thm.pretty_thm ctxt)) thm))
  ];

(* pretty_facts: Proof.context -> bool -> int -> ((string * int) * thm) list -> Pretty.T *)
fun pretty_facts ctxt debug lvl categ_name named_thms =
  let
    fun jsonify fact = Pretty.enclose "{" "}"
      [Print.pretty_json_elem "thm"
        (Print.pretty_json_attribs false "{" "}" (~2) (pretty_fact ctxt fact))]
    val prts = map jsonify named_thms
  in Print.pretty_json_elem categ_name (Print.pretty_json_attribs debug "[" "]" lvl prts) end;

(* pretty_list: ('record -> 'a) -> ('a -> Pretty.T list) 
  -> bool -> int -> string -> 'record -> Pretty.T *)
fun pretty_list retriever prettyfier debug lvl name record =
  let
    fun wrap elem = Print.pretty_json_attribs false "{" "}" 0 [elem]
    val jattribs = Print.pretty_json_attribs debug "[" "]" lvl (map wrap (prettyfier (retriever record)))
  in Print.pretty_json_elem name jattribs end;

in 

(* bool -> int -> _ -> Data.T -> Pretty.T *)
fun pretty_action data = 
  let
    val backslash = "\\";
    val new_line = "\n";
    val quote = "\"";
    val to_write = 
      Data.action_of data
      |> Actions.text_of
      |> Ops.replace_all backslash (backslash ^ backslash) (* for LaTeX comments *)
      |> Ops.replace_all new_line "\\n" (* need to write explicitly "\\n" *)
      |> Ops.replace_all quote (backslash ^ quote)
  in Print.pretty_json_elem "action" (Pretty.quote (Pretty.str to_write)) end;

fun pretty_user_state state =
  let
    val pretty_state = 
      Toplevel.pretty_state state
      |> Pretty.separate " " 
      |> Pretty.block
      |> Pretty.unbreakable
  in Print.pretty_json_elem "user_state" (Pretty.quote pretty_state) end;

fun pretty_goal ctxt data = 
  Print.pretty_json_elem "term" (Pretty.quote (Pretty.unbreakable (Syntax.pretty_term ctxt (Data.goal_of data))));

fun pretty_hyps debug lvl ctxt = 
  pretty_list Data.hyps_of (pretty_terms ctxt) debug lvl "hyps";

fun pretty_proven debug lvl ctxt = 
  pretty_list Data.proven_of (pretty_terms ctxt) debug lvl "proven";

fun pretty_consts debug lvl ctxt = 
  pretty_list Data.consts_of (pretty_consts_pre ctxt) debug lvl "constants";

fun pretty_vars debug lvl ctxt = 
  pretty_list Data.vars_of (pretty_vars_pre ctxt) debug lvl "variables";

fun pretty_types debug lvl ctxt = 
  pretty_list Data.types_of (pretty_types_pre ctxt) debug lvl "type variables";

fun pretty_apply_kwrds debug lvl =
  let
    fun retriever env =
      let
        val cmnds = Data.grp_cmmnds_of env
        val is_apply = Seps.is_apply_step cmnds orf Pred.is "proof"
      in filter is_apply (maps snd cmnds) end;
  in pretty_list retriever pretty_keywords debug lvl "apply_kwrds" end;

fun pretty_isar_kwrds debug lvl =
  let
    fun retriever env =
      let
        val cmnds = Data.grp_cmmnds_of env
        val is_apply = Seps.is_isar_step cmnds
      in filter is_apply (maps snd cmnds) end;
  in pretty_list retriever pretty_keywords debug lvl "isar_kwrds" end; 

fun pretty_methods debug lvl = 
  pretty_list Data.mthds_of pretty_keywords debug lvl "methods";

fun pretty_deps debug lvl ctxt env =
  pretty_facts ctxt debug lvl "deps" (Data.deps_of env)

(* step: bool -> int -> data -> Pretty.T *)
fun pretty_step debug lvl ctxt0 data = 
  let
    val state = Data.state_of data;
    val ctxt0' = if Toplevel.is_proof state 
      then Proof.context_of (Toplevel.proof_of state) 
      else ctxt0
    val ctxt = 
      ctxt0'
      |> Config.put show_sorts false
      |> Config.put show_types false 
    val goal = pretty_goal ctxt data;
    val hs = pretty_hyps debug (lvl+4) ctxt data;
    val seen = pretty_proven debug (lvl+4) ctxt data;
    val cs = pretty_consts debug (lvl+4) ctxt data;
    val vs = pretty_vars debug (lvl+4) ctxt data;
    val Ts = pretty_types debug (lvl+4) ctxt data;
    val usr_out = pretty_user_state state;
    val act = pretty_action data;
    val pretty_state = Print.pretty_json_attribs debug "{" "}" (lvl+2) 
      [act, usr_out, goal, hs, seen, vs, cs, Ts]
  in Print.pretty_json_attrib debug lvl "step" pretty_state end;

(* pretty_json_proof: bool -> int -> Data.env -> Data.T Sections.T -> Pretty.T *)
fun pretty_proof debug lvl env prf =
  let
    val ctxt = Data.ctxt_of env
    val trace = Sections.recover_trace prf;
    val prt_steps =
      map (pretty_step debug (lvl+4) ctxt) trace
      |> Print.pretty_json_attribs debug "[" "]" (lvl+4)
      |> Print.pretty_json_elem "steps"
    val methods = pretty_methods debug (lvl+4) env;
    val ds = pretty_deps debug (lvl+4) ctxt env;
    val applys = pretty_apply_kwrds debug (lvl+4) env;
    val isars = pretty_isar_kwrds debug (lvl+4) env;
    val prty_prf = Print.pretty_json_attribs debug "{" "}" (lvl+2) 
      [prt_steps, applys, isars, methods, ds];
  in Print.pretty_json_attrib debug lvl "proof" prty_prf end;

(* json_proof: (Data.env * Data.T Sections.T) -> string *)
fun json_proof (env, prf) =
  let
    val backslash = "\\";
    val start_quote = backslash ^ "<";
    fun rewrite_quote txt = 
      Ops.replace_all start_quote (backslash ^ start_quote) txt
    val prf = 
      pretty_proof true 0 env prf
      |> Print.string_of_pretty
      |> rewrite_quote
  in prf end;

end;

end;
