
signature TEMP =
sig
  (* QUERYING THEOREMS *)
  val is_app: term -> bool
  val sub_apps_of: term -> term list
  val replace_vars: term -> term
  val generate_queries: Proof.context -> term -> (bool * string Find_Theorems.criterion) list list
  val query_thms: Proof.context -> int -> thm -> (Thm_Name.T * thm) list
  
  (* FINDING NEW THEOREMS *)
  val new_thms: Toplevel.state -> Toplevel.state -> (string * thm list) list

  (* PRETTY PRINTING *)
  val pretty_named_thm: Proof.context -> (string * int) * thm -> Pretty.T list
  val pretty_named_thms: Proof.context -> int -> ((string * int) * thm) list -> Pretty.T
end;


structure Temp: TEMP =
struct


(** QUERYING THEOREMS **)

(* term -> bool *)
fun is_app (_ $ _) = true
  | is_app _ = false

(* term -> term list *)
fun sub_apps_of (Const ("HOL.Trueprop", _) $ t) = sub_apps_of t
  | sub_apps_of (t1 $ t2) = 
    let
      fun cond_add_app t = if is_app t then sub_apps_of t else []
    in  t1 $ t2 :: (cond_add_app t1) @ (cond_add_app t2) end
  | sub_apps_of _ = [];

(* term -> term *)
fun replace_vars (t1 $ t2) = replace_vars t1 $ replace_vars t2
  | replace_vars (Abs (x,T,t)) = Abs (x,T, replace_vars t)
  | replace_vars (Var ((_,i),T)) = Var (("_dummy_", i), T)
  | replace_vars x = x;

(* Proof.context -> term -> (bool * string Find_Theorems.criterion) list list *)
(* TODO: extract query_term *)
fun generate_queries ctxt t =
  let
    val substs = map (fn v => (v, replace_vars v)) (filter is_Var (Get.vars [t]))
    val trms = sub_apps_of t;
    fun query_term t = t
      |> Term.subst_free substs
      |> Term.show_dummy_patterns
      |> Syntax.pretty_term ctxt
      |> Pretty.quote
      |> Print.string_of
      |> Find_Theorems.read_query Position.none
  in map query_term trms end;

(* context -> int -> thm -> (Thm_Name.T * thm) list *)
(* ISSUE: list of theorems is too long *)
fun query_thms ctxt n goal =
  let
    val t = Thm.prop_of goal;
    val find = Find_Theorems.find_theorems_cmd ctxt (SOME goal) (SOME n) true;
    val queries = generate_queries ctxt t;
    fun add_thms q nthms = 
      let
        val results = snd (find q);
        val names = map fst nthms;
        fun in_names (name_cnt, _) = member (op =) names name_cnt
      in fold (Data_Ops.insert_if (Pred.neg in_names)) results nthms end;
  in fold add_thms queries [] end;


(** FINDING NEW THEOREMS **)

(* Toplevel.state -> Toplevel.state -> (string * thm list) list *)
(* TODO: test further *)
fun new_thms state1 state2 =
  if not (Toplevel.is_toplevel state1) andalso not (Toplevel.is_toplevel state2) then
  let
    val generic1 = Toplevel.generic_theory_of state1;
    val generic2 = Toplevel.generic_theory_of state2;
    val facts1 = Context.cases Global_Theory.facts_of Proof_Context.facts_of generic1;
    val facts2 = Context.cases Global_Theory.facts_of Proof_Context.facts_of generic2;
    fun is_new (name, _) = (case Facts.lookup generic1 facts1 name of
        SOME _ => false 
      | NONE => true)
  in Facts.fold_static (Data_Ops.insert_if is_new) facts2 [] end
  else [];

(* Toplevel.state -> (int * Actions.T * Toplevel.state) list 
  -> (string * (int * Actions.T * Toplevel.state * (string * thm list) list) list) list *)
(* ISSUE: too slow and error prone *)
(* fun tag_enum_steps state nass = 
  let
    val acts = map (#2) nass;
    fun lemma_diffs (n:int, a, s) nasfs = (n, a, s, new_thms (#3 nasfs) s);
    val nasfs = Data_Ops.log_fold lemma_diffs (~1, Actions.null, #3 (hd nass), []) nass;
    val thy0 = Toplevel.theory_of (fst (Actions.apply (hd (filter Actions.begins_thy acts)) state));
    fun starts_prf (_, a, _, _) = Actions.begins_proof thy0 a;
    fun ends_prf (_, a, _, _) = Actions.ends_proof thy0 a;
    fun txt_of (_, a, _, _) = Actions.text_of a;
  in Data_Ops.chop_and_tag "proof" txt_of starts_prf ends_prf nasfs end; *)


(* PRETTY PRINTING *)

(* Proof.context -> (string * int) * thm -> Pretty.T list *)
(* TODO: rename and move *)
fun pretty_named_thm ctxt ((name, i),thm) = [
  Print.jpair "name" (
    Pretty.quote (Pretty.block [
      Pretty.marks_str (#1 (Proof_Context.markup_extern_fact ctxt name), name),
      Pretty.str (Thm_Name.print_suffix (name, i))])
  ),
  Print.jpair "term" (Pretty.quote (Thm.pretty_thm ctxt thm))
  ];

(* Proof.context -> int -> ((string * int) * thm) list -> Pretty.T *)
(* TODO: rename and move *)
fun pretty_named_thms ctxt lvl named_thms =
  let
    val prts = map (fn named_thm => 
      Print.jpair "thm"
        (Print.jattribs false "{" "}" (~2) (pretty_named_thm ctxt named_thm))) named_thms
  in Print.jpair "thms" (Print.jattribs true "[" "]" lvl prts) end;

end;
