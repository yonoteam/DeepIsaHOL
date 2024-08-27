
signature TEMP =
sig
  (* QUERYING THEOREMS *)
  val is_app: term -> bool
  val sub_apps_of: term -> term list
  val replace_vars: term -> term
  val generate_queries: Proof.context -> term -> (bool * string Find_Theorems.criterion) list list
  val query_thms: Proof.context -> int -> term -> (Thm_Name.T * thm) list
  val query_thms_goal: Proof.context -> int -> thm -> (Thm_Name.T * thm) list
end;


structure Temp: TEMP =
struct


(** QUERYING THEOREMS **)

(* term -> bool *)
fun is_app (_ $ _) = true
  | is_app _ = false

(* term -> term list *)
fun sub_apps_of (Const ("HOL.Trueprop", _) $ t) = sub_apps_of t
  | sub_apps_of (Const ("Pure.imp", _) $ t) = sub_apps_of t
  | sub_apps_of (Const ("Pure.conjunction", _) $ t) = sub_apps_of t
  | sub_apps_of (t1 $ t2) = 
    let
      fun cond_add_app t = if is_app t then sub_apps_of t else []
    in  t1 $ t2 :: (cond_add_app t1) @ (cond_add_app t2) end
  | sub_apps_of _ = [];

(* term -> term *)
fun replace_vars (t1 $ t2) = replace_vars t1 $ replace_vars t2
  | replace_vars (Abs (x,T,t)) = Abs (x,T, replace_vars t)
  | replace_vars (Var ((_,i),T)) = Var (("_dummy_", i), T)
  | replace_vars (Free (_,T)) = Var (("_dummy_", 0), T)
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
      |> Print.string_of_pretty
      |> Find_Theorems.read_query Position.none
  in map query_term trms end;

(* context -> int -> thm -> (Thm_Name.T * thm) list *)
(* ISSUE: list of theorems is too long *)
fun query_thms ctxt n t =
  let
    val find = Find_Theorems.find_theorems_cmd ctxt NONE (SOME n) true;
    val queries = generate_queries ctxt t;
    fun add_thms q nthms = 
      let
        val results = snd (find q);
        val names = map fst nthms;
        fun in_names (name_cnt, _) = member (op =) names name_cnt
      in fold (Ops.insert_if (Pred.neg in_names)) results nthms end;
  in fold add_thms queries [] end;

(* context -> int -> thm -> (Thm_Name.T * thm) list *)
(* ISSUE: list of theorems is too long *)
fun query_thms_goal ctxt n goal =
  let
    val t = Thm.prop_of goal;
    val find = Find_Theorems.find_theorems_cmd ctxt (SOME goal) (SOME n) true;
    val queries = generate_queries ctxt t;
    fun add_thms q nthms = 
      let
        val results = snd (find q);
        val names = map fst nthms;
        fun in_names (name_cnt, _) = member (op =) names name_cnt
      in fold (Ops.insert_if (Pred.neg in_names)) results nthms end;
  in fold add_thms queries [] end;


(** FINDING PARENTS OF A THEORY **)

(* get_parents: string -> theory list *)
fun get_parents thy_file =
  let
    fun header_from dir =
      let
        val thy_text = Ops.read_file dir;
        val header = Thy_Header.read Position.start thy_text;
      in header end;
    fun theory_from (import_name, _) = 
      (case Basics.try Thy_Info.get_theory import_name of
        SOME thy => thy
        | NONE => 
          (case Resources.find_theory_file import_name of
            SOME (path: Path.T) =>
              let
                val dir = Path.implode path;
              in Resources.begin_theory path (header_from dir) (get_parents dir) end
            | NONE => 
              let 
                val dir = import_name ^ ".thy";
                val path = Path.explode dir;
              in (case Basics.try get_parents dir of 
                  SOME thys => Resources.begin_theory path (header_from dir) thys
                  | NONE => raise Fail ("get_parents: not found " ^ dir)) 
              end
          ))
    val result = List.map theory_from (#imports (header_from thy_file));
  in result end;

end;
