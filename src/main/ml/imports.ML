(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Graph representation of a theory's imports and its locations.
*)

structure Path_Graph = Graph(type key = Path.T val ord = string_ord o apply2 Path.implode);

signature IMPORTS =
sig
  val init: Path.T list -> theory option Path_Graph.T
  val load_upto: Path_Graph.key -> (theory option) Path_Graph.T -> (theory option) Path_Graph.T
  val load_all: theory option Path_Graph.T -> theory option Path_Graph.T
  val to_local_list: Path.T list -> 'a Path_Graph.T -> Path.T list
  val get_init_thy: theory option Path_Graph.T -> Path.T -> theory
  val get_end_thy: theory option Path_Graph.T -> Path.T -> theory
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

(* find_import: Path.T list -> string -> Path.T *)
fun find_import paths import_str = 
  case Resources.find_theory_file import_str of
    SOME path => path
    | NONE => (case locate_in paths import_str of
      SOME path => path
      | NONE => raise Fail ("Could not find path for: " ^ import_str))

in

(* init: Path.T list -> theory option Path_Graph.T *)
fun init paths =
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
        (* val _ = tracing "Found imports" *)
        fun add_next parent graph = 
          let 
            val parent_key = fst parent; 
            (* val _ = tracing ("Attempting " ^ Path.implode parent_key) *)
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
    val opt_thyN = Get.final_thy thy0 thy_text;
  in Path_Graph.map_node thy_path (fn _ => opt_thyN) graph end;

in

(* load_upto: Path_Graph.key -> (theory option) Path_Graph.T -> (theory option) Path_Graph.T *)
fun load_upto thy_path graph =
  if not (Path_Graph.defined graph thy_path)
  then raise Fail ("Imports.load_upto: not a valid import: " ^ Path.implode thy_path)
  else if is_some (Path_Graph.get_node graph thy_path) then graph 
  else let
    val ancesters = List.rev (Path_Graph.all_succs graph [thy_path]);
    fun load_next ancester graph =
      if is_some (Path_Graph.get_node graph ancester) then graph
      else load_one graph ancester;
  in fold load_next ancesters graph end;

end

(* load_all: theory option Path_Graph.T -> theory option Path_Graph.T *)
fun load_all graph =
  let 
    val eldests = Path_Graph.maximals graph;
    val ordered_keys = Path_Graph.all_preds graph eldests;
  in fold load_upto ordered_keys graph end;

(* to_local_list: Path.T list -> 'a Path_Graph.T -> Path.T list *)
fun to_local_list paths graph =
  let
    fun eq_keys (p1, p2) = Pred.is (Path.implode p1) (Path.implode p2)
    val eldests = Path_Graph.maximals graph;
    val ordered_keys = Path_Graph.all_preds graph eldests;
  in Library.inter eq_keys paths ordered_keys end;


local

fun checking_inputs_do graph path f = 
  let val file_name = Path.implode path in
  if not (File.exists path) 
  then raise Fail ("Imports.check: file does not exist: " ^ file_name)
  else if not (Path_Graph.defined graph path)
  then raise Fail ("Imports.check: invalid import: " ^ file_name)
  else f end;

in

(* get_init_thy: theory option Path_Graph.T -> Path.T -> theory *)
fun get_init_thy graph thy_path =
  checking_inputs_do graph thy_path
  (let
    val thy_text = File.read thy_path;
    val header = Thy_Header.read Position.none thy_text;
    val graph' = load_upto thy_path graph;
    val thy0 = 
      Path_Graph.immediate_succs graph' thy_path
      |> map (the o Path_Graph.get_node graph')
      |> Resources.begin_theory (Path.dir thy_path) header;
  in thy0 end);

(* get_end_thy: theory option Path_Graph.T -> Path.T -> theory *)
fun get_end_thy graph thy_path =
  checking_inputs_do graph thy_path 
  (let
    val graph' = load_upto thy_path graph;
  in the (Path_Graph.get_node graph' thy_path) end);

end

end;