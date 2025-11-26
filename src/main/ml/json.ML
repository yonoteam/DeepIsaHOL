(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Internal representation of JSONs as association lists.
*)

signature JSON =
sig
  datatype jtype = 
    JNull
    | JBool of bool
    | JString of string
    | JInt of int
    | JFloat of real
    | JList of jtype list
    | JObj of (string * jtype) list
  type T
  val build: (string * jtype) list -> T
  val build': string -> jtype -> T
  val get: string -> T -> jtype option
  val get_or_else: string -> (jtype -> 'a) -> 'a -> T -> 'a
  val make_jtype_list: string -> ('a -> jtype) -> 'a list -> jtype list
  val upd: string * jtype -> T -> T
  val del: string -> T -> T
  val to_string: T -> string
end;

structure Json: JSON =
struct

datatype jtype = 
  JNull
  | JBool of bool
  | JString of string
  | JInt of int
  | JFloat of real
  | JList of jtype list
  | JObj of (string * jtype) list

datatype T = JSON of (string * jtype) list

fun build js = JSON js;

fun build' name jt = JSON [(name, jt)];

fun get key (JSON js) = AList.lookup (op =) js key;

fun upd (key, value) (JSON js) = JSON (AList.update (op =) (key, value) js);

fun del key (JSON js) = JSON (AList.delete (op =) key js);

fun get_or_else key to_val default (JSON js) =
  (case get key (JSON js) of
    SOME v => to_val v
    | NONE => default);

fun make_jtype_list name to_jtyp vs =
  map (fn v => JObj [(name, to_jtyp v)]) vs;

fun str_of_list _ _ [] = ""
  | str_of_list to_str i [x] = Symbol.spaces i ^ (to_str x)
  | str_of_list to_str i (x :: xs) = 
      Symbol.spaces i ^ (to_str x) ^ ",\n" ^ 
      str_of_list to_str i xs;

fun str_of_key_val to_str (k,v) = Library.quote k ^ ": " ^ to_str v

fun str_of_jtype _ JNull = "null"
  | str_of_jtype _ (JBool b) = Value.print_bool b
  | str_of_jtype _ (JString str) = Print.make_parseable str
  | str_of_jtype _ (JInt n) = Value.print_int n
  | str_of_jtype _ (JFloat r) = Value.print_real r
  | str_of_jtype i (JList xs) = (case xs of [] => "[]"
    | _ => "[\n" ^ str_of_list (str_of_jtype (i+2)) (i+2) xs ^ "\n" ^ Symbol.spaces i ^ "]")
  | str_of_jtype i (JObj js) = (case js of [] => "{}"
    | _ => "{\n" ^ str_of_list (str_of_key_val (str_of_jtype (i+2))) (i+2) js ^ "\n" ^ Symbol.spaces i ^ "}");

fun to_string (JSON js) = str_of_jtype 0 (JObj js);

val _ = ML_system_pp (fn _ => fn _  => ML_Pretty.str o to_string);

end;
