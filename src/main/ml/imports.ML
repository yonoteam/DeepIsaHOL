(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Graph representation of a theory's imports and its locations.
The graph keys are paths to .thy files. The values are pairs of theory's `thy` and bool's `b`. 
If b then thy is the theory at the end of each file, otherwise, it is at the beginning.
*)

structure Path_Graph = Graph(type key = Path.T val ord = string_ord o apply2 Path.implode);

signature IMPORTS =
sig
  (* LOCATING *)
  val locate_in: Path.T list -> string -> Path.T option
  val locate_as_file: string -> Path.T option
  val isa_find_import: string -> Path.T option
  val isa_can_load_thy: string -> bool
  val isa_load_import: string -> Path.T option
  val find_import: Path.T * Path.T list -> string -> Path.T option
  val get_work_path: unit -> Path.T
  val to_string: unit -> string
  val init_for: Path.T -> unit
  val load_upto_start_of: Path.T -> unit
  val load_upto_end_of: Path.T -> unit
  val load_all: unit -> unit
  val start_thy_of: Path.T -> theory
  val final_thy_of: Path.T -> theory
  val list_all: unit -> Path.T list
  val to_local_list: unit -> Path.T list
  val is_at_end: Path.T -> bool
end;

structure Imports: IMPORTS =
struct


(* IMPORTS TYPE *)

type entry = {theory: theory, at_end: bool};

fun make_entry thy at_end = {theory=thy, at_end=at_end}: entry;

type T = {
  work_path: Path.T,
  thy_paths: Path.T list,
  graph: entry option Path_Graph.T
  } Synchronized.var

fun work_path_of (imports: T) = #work_path (Synchronized.value (imports));

fun string_of imports = "Imports(" ^ (Path.implode (work_path_of imports)) ^ ")"

val abs_imports: T = Synchronized.var "Imports_Graph" {
    work_path = File.absolute_path Path.current,
    thy_paths = []: Path.T list,
    graph = Path_Graph.empty: entry option Path_Graph.T
  };

(* val _ = ML_system_pp (fn _ => fn _ => ML_Pretty.str o string_of); *)
fun to_string () = string_of abs_imports;
fun get_work_path () = work_path_of abs_imports;
fun get_thys () = #thy_paths (Synchronized.value abs_imports);
fun get_graph () = #graph (Synchronized.value abs_imports);

fun change_to new_rep = Synchronized.change abs_imports (fn _ => new_rep);

fun upd_graph g = change_to {
  work_path = #work_path (Synchronized.value abs_imports),
  thy_paths = #thy_paths (Synchronized.value abs_imports),
  graph = g};



(* LOCATING *)

local

fun looks_pathy import_str = Pred.contains "/" import_str;

fun looks_namy import_str = Pred.contains "." import_str 
  andalso not (looks_pathy import_str)
  andalso not (Pred.ends_with ".thy" import_str);

fun is_thy_path import_str = looks_pathy import_str 
  andalso Pred.ends_with ".thy" import_str;

fun get_file_name import_str =
  if looks_namy import_str then (Long_Name.base_name import_str) ^ ".thy"
  else if is_thy_path import_str then Path.implode (Path.base (Path.explode import_str))
  else if looks_pathy import_str then Path.implode (Path.base (Path.explode (import_str ^ ".thy")))
  else if Pred.ends_with ".thy" import_str then import_str
  else import_str ^ ".thy";

in 

fun locate_in paths import_str =
  let
    val possible_ending = get_file_name import_str;
    fun ends_as_import path = Pred.ends_with possible_ending (Path.implode path)
    val opt_result = Pred.find_first ends_as_import paths;
  in opt_result end;

fun locate_as_file import_str =
  let val file_path = Path.explode (import_str ^ ".thy") in
  if File.is_file file_path
  then SOME file_path
  else NONE end;

end;

fun isa_find_import import_str = 
  if Basics.can Resources.find_theory_file import_str
  then Resources.find_theory_file import_str
  else NONE;

val isa_can_load_thy = Basics.can Thy_Info.get_theory;

fun isa_load_import import_str =
  if isa_can_load_thy import_str 
  then SOME (Path.explode ("ISA_FOUND="^import_str))
  else NONE;

local

fun directory_has_thy path = File.read_dir path
  |> List.exists (Pred.ends_with ".thy");

fun find_root_file path =
  let
    val dir_path = if File.is_dir path then path else Path.dir path
    val root_candidate = Path.append dir_path (Path.basic "ROOT")
  in
    if File.is_file root_candidate then SOME root_candidate
    else
      if not (directory_has_thy dir_path) then NONE
      else
        let
          val parent = Path.dir dir_path
        in 
          if File.eq (dir_path, parent) then NONE
          else find_root_file parent
        end
  end;

fun find_sessions root_file = File.read root_file
  |> Ops.split_at "session "
  |> Pred.drop_until (Pred.contains "session ")
  |> filter_out (Pred.is "session ")
  |> map (hd o (Library.space_explode " ") o (Ops.replace_all "\"" ""));

fun maybe_find_in importer_path import_str =
  let
    val importer_dir = Path.dir importer_path;
    val imported_path = Path.append importer_dir (Path.explode (import_str ^ ".thy"));
  in if File.is_file imported_path then SOME imported_path else NONE end

in

fun locate_via_importer importer_path import_str =
  case find_root_file importer_path of
    NONE => maybe_find_in importer_path import_str
  | SOME root_path =>
    let
      val sessions = find_sessions root_path
      val results = sessions
        |> map (fn name => isa_find_import (name ^ "." ^ import_str))
        |> filter is_some
      val final = 
        if null results then maybe_find_in importer_path import_str
        else if length results > 1 
        then
          let
            (* val _ = map (fn opt_p => Output.tracing ("found double:" ^ (Path.implode (the opt_p)))) results *)
            fun shares_importer_dir opt_p = 
              String.isSubstring (Path.implode (Path.dir importer_path)) (Path.implode (the opt_p));
            val shared = filter shares_importer_dir results;
          in if null shared orelse length shared > 1 then 
            let
              val part1 = "Imports.locate_via_session_root: found two candidate sessions for ";
              val part2 = " and '"
              val part3 = "'. Returning first one."
              val mssg = part1 ^ import_str ^ part2 ^ (Path.implode importer_path) ^ part3
            in (Output.warning mssg); hd results end
            else hd shared end
        else hd results
    in final end;

end;

infix 1 else_try;

fun (opt_x else_try opt_y) =
  (case opt_x of SOME y => SOME y | NONE => opt_y); 

fun find_import (importer, paths) import_str =
  isa_load_import import_str
  else_try isa_find_import import_str
  else_try locate_via_importer importer import_str
  else_try locate_in paths import_str
  else_try locate_as_file import_str;


(* INITIALIZATION *)

fun init_found found_str =
  let
    val key = Library.unprefix "ISA_FOUND=" found_str;
    val thy = Thy_Info.get_theory key;
  in (the (isa_find_import key), SOME (make_entry thy true)) end;

fun init_entry (importer, paths) import_str =
  (case find_import (importer, paths) import_str of
    SOME path => 
      let val path_str = Path.implode path in 
      if Pred.starts_with "ISA_FOUND=" path_str
      then init_found path_str
      else (path, NONE) end
    | NONE => 
      (Output.warning ("Did not find import: " ^ import_str);
      (Path.explode import_str, NONE))
  )

fun init_deps paths =
  let
    fun add_ancesters path graph =
      if is_some (Path_Graph.get_node graph path) then graph
      else let
        (* val _ = Output.tracing ("Adding path " ^ (Path.implode path)); *)
        val imports = path
          |> File.read
          |> Thy_Header.read Position.none
          |> #imports
          |> map (init_entry (path, paths) o fst);
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

fun init_for path =
  let
    val thy_paths = Pred.find_files (Pred.ends_with ".thy" o Path.implode) path;
    val graph = init_deps thy_paths;  
  in change_to {work_path=path, thy_paths=thy_paths, graph=graph} end;


(* LOADING *)

fun final_thy thy0 thy_text =
  let
    val transitions =  
      thy_text
      |> Outer_Syntax.parse_text thy0 (K thy0) Position.start
      |> filter_out (fn tr => Toplevel.name_of tr = "<ignored>");
    val final_state = 
      Toplevel.make_state NONE
      |> fold (Toplevel.command_exception true) transitions;
  in Toplevel.previous_theory_of final_state end;

(* it assumes that all parent theories are defined in the graph *)
fun upd_with_start_thy thy_path graph =
  let
    val thy_text = File.read thy_path;
    val header = Thy_Header.read Position.none thy_text;
    val parents = 
      Path_Graph.immediate_succs graph thy_path
      |> map (#theory o the o Path_Graph.get_node graph)
    val thy0 = Resources.begin_theory (Path.dir thy_path) header parents;
  in Path_Graph.map_node thy_path (fn _ => SOME (make_entry thy0 false)) graph end;

(* it assumes that all parent theories are defined in the graph *)
fun upd_with_final_thy thy_path graph =
  let
    val thy_text = File.read thy_path;
    val header = Thy_Header.read Position.none thy_text;
    val parents = 
      Path_Graph.immediate_succs graph thy_path
      |> map (#theory o the o Path_Graph.get_node graph)
    val thy0 = Resources.begin_theory (Path.dir thy_path) header parents;
    val new_entry = (case final_thy thy0 thy_text of
      SOME thy => SOME (make_entry thy true)
      | NONE => NONE);
  in Path_Graph.map_node thy_path (fn _ => new_entry) graph end;

fun upd_ancesters_of thy_path graph =
  if not (Path_Graph.defined graph thy_path)
  then raise Fail ("Imports.upd_ancesters_of: not a valid import: " ^ Path.implode thy_path)
  else let
    val ancesters = List.rev (tl (Path_Graph.all_succs graph [thy_path]));
    fun load_next ancester graph =
      (case Path_Graph.get_node graph ancester of
        SOME entry => if #at_end entry then graph else upd_with_final_thy ancester graph
        | NONE => upd_with_final_thy ancester graph);
  in fold load_next ancesters graph end;

fun upd_upto_start_of thy_path graph =
  upd_ancesters_of thy_path graph
  |> upd_with_start_thy thy_path;

fun upd_upto_end_of thy_path graph =
  upd_ancesters_of thy_path graph
  |> upd_with_final_thy thy_path;

fun load_upto_end_of thy_path =
  upd_graph (upd_upto_end_of thy_path (get_graph ()));

fun load_upto_start_of thy_path =
  upd_graph (upd_upto_start_of thy_path (get_graph ()));

fun all_keys graph =
  let
    val eldests = Path_Graph.maximals graph;
    val ordered_keys = Path_Graph.all_preds graph eldests;
  in ordered_keys end;

fun upd_all graph =
  fold upd_upto_end_of (all_keys graph) graph;

val load_all = upd_graph o upd_all o get_graph;

fun checking_inputs_of_do path f = 
  let val file_name = Path.implode path in
  if not (File.exists path) 
  then raise Fail ("Imports.check: file does not exist: " ^ file_name)
  else if not (Path_Graph.defined (get_graph ()) path)
  then raise Fail ("Imports.check: invalid import: " ^ file_name)
  else f end;

fun start_thy_of thy_path =
  checking_inputs_of_do thy_path
  (load_upto_start_of thy_path; #theory (the (Path_Graph.get_node (get_graph ()) thy_path)));

fun final_thy_of thy_path =
  checking_inputs_of_do thy_path 
  (load_upto_end_of thy_path; #theory (the (Path_Graph.get_node (get_graph ()) thy_path)));


(* INFORMATION RETRIEVAL *)

fun list_all () = all_keys (get_graph ());

fun to_local_list () =
  let
    fun eq_keys (p1, p2) = Pred.is (Path.implode p1) (Path.implode p2)
    val ordered_keys = all_keys (get_graph ());
  in Library.inter eq_keys (get_thys ()) ordered_keys end;

fun is_at_end thy_path =
  checking_inputs_of_do thy_path 
    (case Path_Graph.get_node (get_graph ()) thy_path of
      SOME entry => #at_end entry
      | NONE => false);

end;