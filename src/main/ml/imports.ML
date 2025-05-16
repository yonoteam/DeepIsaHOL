(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Graph representation of a theory's imports and its locations.
*)

signature IMPORTS =
sig
  type T
  val get_work_path: unit -> Path.T
  val to_string: unit -> string
  val init_for: Path.T -> unit
  val load_upto: Path.T -> unit
  val load_all: unit -> unit
  val to_local_list: unit -> Path.T list
  val init_thy_of: Path.T -> theory
  val end_thy_of: Path.T -> theory
end;

structure Path_Graph = Graph(type key = Path.T val ord = string_ord o apply2 Path.implode);

structure Imports: IMPORTS =
struct


(* IMPORTS TYPE *)

type T = {
  work_path: Path.T,
  thy_paths: Path.T list,
  graph: (theory option) Path_Graph.T
  } Synchronized.var

fun work_path_of (imports: T) = #work_path (Synchronized.value (imports));

fun string_of imports = "Imports(" ^ (Path.implode (work_path_of imports)) ^ ")"

val abs_imports: T = Synchronized.var "Imports_Graph" {
    work_path = File.absolute_path Path.current,
    thy_paths = []: Path.T list,
    graph = Path_Graph.empty: (theory option) Path_Graph.T
  };

(* val _ = ML_system_pp (fn _ => fn _ => ML_Pretty.str o string_of); *)
fun to_string () = string_of abs_imports;
fun get_work_path () = work_path_of abs_imports;
fun get_thys () = #thy_paths (Synchronized.value abs_imports);
fun get_graph () = #graph (Synchronized.value abs_imports);

(* change_to: Imports.T -> unit *)
fun change_to new_rep = Synchronized.change abs_imports (fn _ => new_rep);

(* upd_graph: (theory option) Path_Graph.T -> unit *)
fun upd_graph g = change_to {
  work_path = #work_path (Synchronized.value abs_imports),
  thy_paths = #thy_paths (Synchronized.value abs_imports),
  graph = g};



(* LOCATION *)

(* locate_in: Path.T list -> string -> Path.T option *)
fun locate_in paths import_str =
  let
    val get_name = Path.base #> Path.implode
    val name_of_import =
      if Pred.contains "." import_str
        andalso not (Pred.contains "/" import_str)
      then (Long_Name.base_name import_str) ^ ".thy"
      else if Pred.contains "/" import_str
      then import_str |> Path.explode |> get_name
      else import_str ^ ".thy"
    fun has_import_name path = Pred.is name_of_import (get_name path)
    val opt_result = Pred.find_first has_import_name paths;
  in opt_result end;

(* find_import: Path.T list -> string -> Path.T *)
fun find_import paths import_str = 
  case Resources.find_theory_file import_str of
    SOME path => path
    | NONE => (case locate_in paths import_str of
      SOME path => path
      | NONE => raise Fail ("find_import: Could not find path for: " ^ import_str));


(* INITIALIZATION *)

(* init_deps: Path.T list -> theory option Path_Graph.T *)
fun init_deps paths =
  let
    fun add_ancesters path graph =
      if is_some (Path_Graph.get_node graph path) then graph
      else let
        fun init_entry import_str =
          case Basics.try Thy_Info.get_theory import_str of
            SOME thy => (Path.explode ("ISA_FOUND=" ^ import_str), SOME thy)
            | NONE => (find_import paths import_str, NONE);
        val imports = path
          |> File.read
          |> Thy_Header.read Position.none
          |> #imports
          |> map (init_entry o fst);
        fun add_next parent graph = 
          let 
            val parent_key = fst parent; 
          in
            if Path_Graph.defined graph parent_key
            then Path_Graph.add_edge (path, parent_key) graph
            else Path_Graph.new_node parent graph
              |> Path_Graph.add_edge (path, parent_key)
              |> add_ancesters parent_key
          end;
      in fold add_next imports graph end;
    fun do_next path graph =
      if Path_Graph.defined graph path then graph
      else let
        val graph' = Path_Graph.new_node (path, NONE) graph;
      in add_ancesters path graph' end;
  in fold do_next paths Path_Graph.empty end;

(* init_for: Path.T -> unit *)
fun init_for path =
  let
    val thy_paths = Pred.find_files (Pred.ends_with ".thy" o Path.implode) path;
    val graph = init_deps thy_paths;
  in change_to {work_path=path, thy_paths=thy_paths, graph=graph} end;


(* LOADING *)

(* load_latest: (theory option) Path_Graph.T -> Path.T -> (theory option) Path_Graph.T *)
(* it assumes that all parent theories are defined in the graph *)
fun load_latest graph thy_path =
  let
    val thy_text = File.read thy_path;
    val header = Thy_Header.read Position.none thy_text;
    val parents = 
      Path_Graph.immediate_succs graph thy_path
      |> map (the o Path_Graph.get_node graph)
    val thy0 = Resources.begin_theory (Path.dir thy_path) header parents;
    val opt_thyN = Get.final_thy thy0 thy_text;
  in Path_Graph.map_node thy_path (fn _ => opt_thyN) graph end;

(* update_upto: Path_Graph.key -> (theory option) Path_Graph.T -> (theory option) Path_Graph.T *)
fun upd_upto thy_path graph =
  if not (Path_Graph.defined graph thy_path)
  then raise Fail ("Imports.load_upto: not a valid import: " ^ Path.implode thy_path)
  else if is_some (Path_Graph.get_node graph thy_path) then graph 
  else let
    val ancesters = List.rev (Path_Graph.all_succs graph [thy_path]);
    fun load_next ancester graph =
      if is_some (Path_Graph.get_node graph ancester) then graph
      else load_latest graph ancester;
  in fold load_next ancesters graph end;

(* load_upto: Path.T -> unit *)
fun load_upto thy_path =
  upd_graph (upd_upto thy_path (get_graph ()));

(* upd_all: theory option Path_Graph.T -> theory option Path_Graph.T *)
fun upd_all graph =
  let 
    val eldests = Path_Graph.maximals graph;
    val ordered_keys = Path_Graph.all_preds graph eldests;
  in fold upd_upto ordered_keys graph end;

(* load_all: unit -> unit *)
val load_all = upd_graph o upd_all o get_graph;

(* to_local_list: Path.T list *)
fun to_local_list () =
  let
    fun eq_keys (p1, p2) = Pred.is (Path.implode p1) (Path.implode p2)
    val eldests = Path_Graph.maximals (get_graph ());
    val ordered_keys = Path_Graph.all_preds (get_graph ()) eldests;
  in Library.inter eq_keys (get_thys ()) ordered_keys end;

fun checking_inputs_do path f = 
  let val file_name = Path.implode path in
  if not (File.exists path) 
  then raise Fail ("Imports.check: file does not exist: " ^ file_name)
  else if not (Path_Graph.defined (get_graph ()) path)
  then raise Fail ("Imports.check: invalid import: " ^ file_name)
  else f end;

(* init_thy_of: Path.T -> theory *)
fun init_thy_of thy_path =
  checking_inputs_do thy_path
  (let
    val thy_text = File.read thy_path;
    val header = Thy_Header.read Position.none thy_text;
    val _ = load_upto thy_path;
    val thy0 = 
      Path_Graph.immediate_succs (get_graph ()) thy_path
      |> map (the o Path_Graph.get_node (get_graph ()))
      |> Resources.begin_theory (Path.dir thy_path) header;
  in thy0 end);

(* end_thy_of: theory option Path_Graph.T -> Path.T -> theory *)
fun end_thy_of thy_path =
  checking_inputs_do thy_path 
  (load_upto thy_path; the (Path_Graph.get_node (get_graph ()) thy_path));

end;