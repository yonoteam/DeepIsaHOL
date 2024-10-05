
signature TEMP =
sig
  (* QUERYING THEOREMS *)
  val is_app: term -> bool
  val sub_apps_of: term -> term list
  val replace_vars: term -> term
  val generate_queries: Proof.context -> term -> (bool * string Find_Theorems.criterion) list list
  val query_thms: Proof.context -> int -> term -> (string * thm) list

  (* DATA WRITING OPS *)
  val write_proof: string -> int -> (Toplevel.state * Actions.T) Sections.T -> unit
  val write_proof': string -> int -> (Toplevel.state * Actions.T) Sections.T -> unit
  val write_proofs: string -> ((Toplevel.state * Actions.T) Sections.T) list -> unit
  val retrieve_from_to: string -> string -> unit
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
        val results = map (Library.apfst Facts.ref_name) (snd (find q));
        val names = map fst nthms;
        fun in_names (name_cnt, _) = member (op =) names name_cnt
      in fold (Ops.insert_if (Pred.neg in_names)) results nthms end;
  in fold add_thms queries [] end;


(** DATA WRITING OPS **)

(* write_proof': string -> int -> (Toplevel.state * Actions.T) Sections.T -> unit *)
fun write_proof' dir i proof =
  let
    (* IO operartions *)
    val env = Data.get_env I proof;
    val thy_name = Context.theory_name {long=false} (Toplevel.theory_of (Data.s1_of env))
    val dir_name = dir ^ "/" ^ thy_name ^ "/" ^ "proof" ^ Value.print_int i;
      (* extra "/"s are no issue for OS.FileSys.mkDir *)
    val _ = Ops.create_directory {force=true} dir_name
        handle Fail txt => Exn.reraise (Fail txt);
    fun fname j = ("step" ^ Value.print_int j ^ ".json");

    (* Data.T operartions *)
    val start = Data.init env (Sections.get_first proof);
    val proof_data = Sections.do_one_then_fold true start (Data.next env) proof
    val trace = Sections.recover_trace proof_data;
    
    (* String operartions *)
    val backslash = "\\";
    val start_quote = backslash ^ "<";
    fun rewrite_quote txt = Ops.replace_all start_quote (backslash ^ start_quote) txt
    
    (* Main *)
    fun write_next data j =
      let
        val _ = Ops.create_file {force=false} dir_name (fname j) "";
        val _ =
          data
          |> Data.pretty_json_step true 0 (Data.ctxt_of env)
          |> Print.string_of_pretty
          |> rewrite_quote
          |> Print.write_to (Print.FILE (OS.Path.joinDirFile {dir=dir_name, file=fname j}))
      in j+1 end;
    val _ = if null trace 
      then raise Fail "Data.write_proof: empty proof" 
      else fold write_next (tl trace) (write_next (hd trace) 0)
  in () end;

(* retrieve_from_to: string -> string -> unit *)
fun retrieve_from_to thy_file write_dir =
  let
    val thy0 = Get.start_thy_from_file (Get.parents thy_file) thy_file;
    val thy_text = Ops.read_file thy_file
  in Data.write_proofs (write_dir, thy0, thy_text) end;

end;
