(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Unfinished graph representation of a theory's imports and its locations.
*)

structure Path_Graph = Graph(type key = Path.T val ord = string_ord o apply2 Path.implode);

signature IMPORTS =
sig
  val init_imports: Path.T list -> theory option Path_Graph.T
  val load_upto: (theory option) Path_Graph.T -> Path_Graph.key -> (theory option) Path_Graph.T
  val locate: 'a Path_Graph.T -> Path.T -> string -> Path.T
  val init_dep_graph: {debug: bool} -> Path.T -> string list Path_Graph.T -> string list Path_Graph.T
  val complete_dep_graph: {debug: bool, ignore_errors:bool} -> string list Path_Graph.T -> string list Path_Graph.T
end;


structure Imports: IMPORTS =
struct

local

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

(* locate: Path.T list -> string -> Path.T *)
fun locate paths import_str = 
  case Resources.find_theory_file import_str of
    SOME path => path
    | NONE => (case locate_in paths import_str of
      SOME path => path
      | NONE => raise Fail ("Could not find path for: " ^ import_str))

in

(* init_imports: Path.T list -> theory option Path_Graph.T *)
fun init_imports paths =
  let
    fun add_ancesters path graph =
      let
        fun init_entry import_str =
          case Basics.try Thy_Info.get_theory import_str of
            SOME thy => (Path.explode ("ISA_FOUND=" ^ import_str), SOME thy)
            | NONE => (locate paths import_str, NONE);
        val imports = path
          |> File.read
          |> Thy_Header.read Position.none
          |> #imports
          |> map (init_entry o fst);
        (* val _ = tracing "Found imports" *)
        fun cond_add key graph =
          if is_some (Path_Graph.get_node graph key) then graph
          else add_ancesters key graph;
        fun add_next parent graph = 
          let 
            val parent_key = fst parent; 
            (* val _ = tracing ("Attempting " ^ Path.implode parent_key) *)
          in
            if Path_Graph.defined graph parent_key
            then if Path_Graph.is_edge graph (path, parent_key) then graph
              else Path_Graph.add_edge (path, parent_key) graph
            else Path_Graph.new_node parent graph
              |> Path_Graph.add_edge (path, parent_key)
              |> cond_add parent_key
          end;
      in fold add_next imports graph end;
    fun do_next path graph =
      let
        val graph' = 
          if Path_Graph.defined graph path
          then graph else Path_Graph.new_node (path, NONE) graph;
        (* val _ = tracing ("Added " ^ Path.implode path) *)
      in add_ancesters path graph' end;
  in fold do_next paths Path_Graph.empty end;

end

local

(* load_one: (theory option) Path_Graph.T -> Path.T -> (theory option) Path_Graph.T *)
(* TODO: add error message for the assumption that all parents should be defined *)
fun load_one graph thy_path =
  let
    val thy_text = File.read thy_path;
    val header = Thy_Header.read Position.none thy_text;
    val parents = 
      Path_Graph.immediate_succs graph thy_path
      |> map (the o Path_Graph.get_node graph)
    val thy0 = Resources.begin_theory (Path.dir thy_path) header parents;
    val thyN = Get.final_thy thy0 thy_text;
  in Path_Graph.map_node thy_path (fn _ => thyN) graph end;

in

fun load_upto graph thy_path =
  let
    val ancesters = List.rev (Path_Graph.all_succs graph [thy_path]);
    fun load_next ancester graph =
      if is_some (Path_Graph.get_node graph ancester) then graph
      else load_one graph ancester;
  in fold load_next ancesters graph end;

end



(*********************)
(* VERSION 2: LEGACY *)
(*********************)
local

infix 1 if_none_then

(* if_none_then: ('a -> 'b option) * ('a -> 'b option) -> 'a -> 'b option *)
fun (f if_none_then g) x = 
  case f x of 
    SOME y => SOME y
    | NONE => g x;

(* locate_via_thy: string -> Path.T option *)
fun locate_via_thy import_str =
  (case Basics.try Thy_Info.get_theory import_str of
    SOME _ => SOME (Path.basic "ISABELLE")
    | NONE => NONE);

(* TODO: Report that it fails if the import_name has "/" *)
(* locate_via_file: string -> Path.T option *)
fun locate_via_file import_name =
  (case Resources.find_theory_file import_name of
    SOME path => SOME path
    | NONE => NONE);

(* locate_via_full_path: Path.T -> string -> Path.T option *)
fun locate_via_full_path work_dir import_name =
  let
    val basic_file_path = Path.explode (import_name ^ ".thy")
    val maybe_full_path = Path.append work_dir basic_file_path;
    val result = 
      if File.is_file maybe_full_path 
      then SOME maybe_full_path
      else if Path.is_absolute basic_file_path 
        andalso File.is_file basic_file_path 
      then SOME basic_file_path
      else NONE
  in result end;

(* locate_in_graph: 'a Path_Graph.T -> string -> Path_Graph.key option *)
fun locate_in_graph graph import_name =
  Path_Graph.restrict (Pred.is (import_name ^ ".thy") o Path.implode o Path.base) graph
  |> Basics.try (the_single o Path_Graph.keys);

in

(* locate: 'a Path_Graph.T -> Path.T -> string -> Path.T *)
fun locate graph work_dir import_name =
  let
    val attempts = locate_via_thy
      if_none_then (locate_via_full_path work_dir)
      if_none_then locate_via_file
      if_none_then (locate_in_graph graph);
    val result = (case attempts import_name of 
     SOME location => location
     | NONE => raise Fail ("get_location: could not find " ^ import_name 
        ^ "(tried in " ^ (Path.implode work_dir) ^ ")"))
  in result end;

end;

local 

(* imports_of: Path.T -> string list *)
fun imports_of thy_path =
  File.read thy_path
  |> Thy_Header.read Position.start
  |> #imports |> map fst;

(* carefully_add: ''a -> Path.T -> ''a list Path_Graph.T -> ''a list Path_Graph.T *)
fun carefully_add new_val path graph = 
  if member (op = o apply2 Path.implode) (Path_Graph.keys graph) path then 
    let
      fun cond_expand vals = 
        if member (op =) vals new_val then vals else new_val :: vals
    in Path_Graph.map_node path cond_expand graph end
  else Path_Graph.new_node (path, [new_val]) graph;

in

(* TODO: Make it of type theory Path_Graph.T and account for theories outside of work_dir not in Isabelle libraries *)
(* init_dep_graph: {debug: bool} -> Path.T -> string list Path_Graph.T -> string list Path_Graph.T *)
fun init_dep_graph {debug=debugging} work_dir dep_graph =
  let
    val _ = if debugging then tracing ("Trying " ^ Path.implode work_dir) else ();
    val working_path = File.check_dir work_dir;
    val dir_contents = File.read_dir working_path;
    val thy_paths = 
      filter (Pred.ends_with ".thy") dir_contents
      |> map (Path.append working_path o Path.explode);
    val dirs =
      filter_out (Pred.contains ".") dir_contents
      |>  map (Path.append working_path o Path.explode)
      |> filter (File.is_dir andf Path.is_absolute);
    fun add_thy_dir thy_path graph =
      let
        val _ = if debugging then tracing ("Processing " ^ Path.implode thy_path) else ();
        val thy_name = Path.implode (fst (Path.split_ext ((Path.base thy_path))))
      in carefully_add thy_name thy_path graph end;
  in fold (init_dep_graph {debug=debugging}) dirs (fold add_thy_dir thy_paths dep_graph) end;

(* TODO: Make less spaguetti. *)
(* complete_dep_graph: {debug: bool} -> string list Path_Graph.T -> string list Path_Graph.T *)
fun complete_dep_graph {debug=debugging, ignore_errors=ignoring_errors} dep_graph =
  let
    val thy_paths = Path_Graph.keys dep_graph;
    fun process_imports thy_path graphA =
      let
        val _ = if debugging then tracing ("Processing " ^ Path.implode thy_path) else ();
        val work_dir = Path.dir thy_path
        fun expand_graph import graph =
          let
            val result = if ignoring_errors
              then let
                val location = Basics.try (locate graph work_dir) import;
                val graph' = (case location of 
                  SOME loc => 
                    carefully_add import loc graph
                    |> Path_Graph.add_edge (thy_path, loc)
                  | NONE => ((if debugging then tracing ("Skipping due to error: " ^ import) else ()); graph))
              in graph' end 
              else let
                val location = locate graph work_dir import;
                val graph' = carefully_add import location graph
                  |> Path_Graph.add_edge (thy_path, location)
              in graph' end;
          in result end;
      in fold expand_graph (imports_of thy_path) graphA end;
  in fold process_imports thy_paths dep_graph end;

end;

end;