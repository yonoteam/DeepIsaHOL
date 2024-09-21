
structure Path_Graph = Graph(type key = Path.T val ord = string_ord o apply2 Path.implode);

signature IMPORTS =
sig
  val locate: 'a Path_Graph.T -> Path.T -> string -> Path.T
  val init_dep_graph: {debug: bool} -> Path.T -> string list Path_Graph.T -> string list Path_Graph.T
  val complete_dep_graph: {debug: bool} -> string list Path_Graph.T -> string list Path_Graph.T
end;


structure Imports: IMPORTS =
struct

local

infix 1 if_none_then

(* if_none_then: ('a -> 'b option) * ('a -> 'b option) -> 'a -> 'b option *)
fun (f if_none_then g) x = (case f x of 
  SOME y => SOME y
  | NONE => g x);

(* locate_via_thy: string -> Path.T option *)
fun locate_via_thy import_name =
  (case Basics.try Thy_Info.get_theory import_name of
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

(* TODO: Create a {unsafe} mode for ignoring malformed subdirectories *)
(* complete_dep_graph: {debug: bool} -> string list Path_Graph.T -> string list Path_Graph.T *)
fun complete_dep_graph {debug=debugging} dep_graph =
  let
    val thy_paths = Path_Graph.keys dep_graph;
    fun process_imports thy_path graphA =
      let
        val _ = if debugging then tracing ("Processing " ^ Path.implode thy_path) else ();
        val work_dir = Path.dir thy_path
        fun expand_graph import graph =
          let
            val location = locate graph work_dir import;
            val graph' = 
              carefully_add import location graph
              |> Path_Graph.add_edge (thy_path, location);
          in graph' end;
      in fold expand_graph (imports_of thy_path) graphA end;
  in fold process_imports thy_paths dep_graph end;

end;

end;