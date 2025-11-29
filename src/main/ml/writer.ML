(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Interface with writer.scala.
*)

signature WRITER =
sig
  val write_json_proof: string -> (Data.env * Data.T Sections.T) -> unit
  val write_json_proofs: (string * theory * string) -> unit
end;


structure Writer: WRITER =
struct

local

fun not_member_cons_if P x xs = Pred.cons_if (P andf (Pred.neg (member (op =) xs))) x xs;

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

in

fun entry_to_jtype name key_to_str val_to_str (i, (k, vs)) =
  let
    val v = (Library.space_implode ", " (map val_to_str vs)) ^ " :: " ^ key_to_str k;
  in (name ^ Value.print_int i, Json.JString v) end;

fun json_step ctxt0 data =
  let
    val st = Data.state_of data;
    val ctxt0' = if Toplevel.is_proof st 
      then Proof.context_of (Toplevel.proof_of st) 
      else ctxt0
    val ctxt = ctxt0'
      |> Config.put show_sorts false
      |> Config.put show_types false;
    val act = Data.action_of data
      |> Actions.text_of;
    val usr_st = Get.user_state {break_lines=false} st;
    val goal = Data.goal_of data
      |> Print.string_of_term ctxt;
    val hyps = Data.hyps_of data
      |> map (fn v => Json.JObj [("term", Json.JString (Print.string_of_term ctxt v))]);
    val proven = Data.proven_of data
      |> map (fn v => Json.JObj [("proven", Json.JString (Print.string_of_term ctxt v))]);
    val vs = Data.vars_of data
      |> alist_by vars_info
      |> Ops.enumerate
      |> map (entry_to_jtype "Type" (Print.string_of_type ctxt) (Print.string_of_term ctxt));
    val cs = Data.consts_of data
      |> alist_by const_info
      |> Ops.enumerate
      |> map (entry_to_jtype "Type" (Print.string_of_type ctxt) (Print.string_of_term ctxt));
    val Ts = Data.types_of data
      |> alist_by types_info
      |> Ops.enumerate
      |> map (entry_to_jtype "Type" (Print.string_of_sort ctxt) (Print.string_of_type ctxt));
  in Json.JObj [
    ("action", Json.JString act),
    ("user_state", Json.JString usr_st),
    ("term", Json.JString goal),
    ("hyps", Json.JList hyps),
    ("proven", Json.JList proven),
    ("variables", Json.JObj vs),
    ("constants", Json.JObj cs),
    ("type variables", Json.JObj Ts)
    ]
  end;

end

fun json_proof (env, prf) =
  let
    val ctxt = Data.ctxt_of env
      |> Config.put show_sorts false
      |> Config.put show_types false;
    val trace = Sections.recover_trace prf;
    val steps = map (json_step ctxt) trace;
    val ms = Data.mthds_of env
      |> map (fn str => Json.JString str);
    
    val cmnds = Data.grp_cmmnds_of env;
    val is_apply = Seps.is_apply_step cmnds orf Pred.is "proof"
    val is_isar = Seps.is_isar_step cmnds;
    val kwrds = maps snd cmnds;
    val applys = filter is_apply kwrds
      |> map (fn str => Json.JString str);
    val isars = filter is_isar kwrds
      |> map (fn str => Json.JString str);
    
    fun wrap_dep ((name, i), thm) = Json.JObj [
      ("name", Json.JString (Thm_Name.print (name, i))), 
      ("term", Json.JString (Print.string_of_pretty (Thm.pretty_thm ctxt thm)))]
    val deps = Data.deps_of env
      |> map wrap_dep
  in Json.build [
    ("proof", Json.JList steps),
    ("apply_kwrds", Json.JList applys),
    ("isar_kwrds", Json.JList isars),
    ("methods", Json.JList ms),
    ("deps", Json.JList deps)
    ]
  end;

fun write_json_proof file_path proof = if Ops.file_exists file_path 
  then warning ("write_proof: File " ^ file_path ^ " already exists (skipping)")
  else json_proof proof |> Json.write_to file_path;

fun write_json_proofs (write_dir, thy0, thy_text) =
  let 
    val proofs = Data.extract_proofs thy0 thy_text;
    fun fname j = ("proof" ^ Value.print_int j ^ ".json");
    fun write_next proof i =
      let
        val file_path = OS.Path.joinDirFile {dir=write_dir, file=fname i}
        val _ = write_json_proof file_path proof
      in i+1 end;
    val _ = fold write_next proofs 0
  in () end;

end;
