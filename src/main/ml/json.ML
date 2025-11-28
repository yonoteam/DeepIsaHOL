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
  val paths: T -> (jtype list * jtype) list
  val fold: (jtype list * jtype -> 'a -> 'a) -> jtype -> 'a -> 'a
  val make_jtype_list: string -> ('a -> jtype) -> 'a list -> jtype list
  val upd: string * jtype -> T -> T
  val del: string -> T -> T
  val size: jtype -> int
  val to_string: T -> string
  val write_to: string -> T -> unit
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

fun str_of_atom JNull = "null"
  | str_of_atom (JBool b) = Value.print_bool b
  | str_of_atom (JInt n) = Value.print_int n
  | str_of_atom (JFloat r) = Value.print_real r
  | str_of_atom (JString s) = Print.make_parseable s
  | str_of_atom _ = raise Fail "Json.str_of_atom called on non-atom.";

fun atoms idxs (JObj [(k, v)]) = atoms (JString k :: idxs) v
  | atoms idxs (JObj ((k, v) :: kvs')) = 
    let
      val atoms1 = atoms (JString k :: idxs) v
      val atoms2 = atoms idxs (JObj kvs')
    in atoms1 @ atoms2 end
  | atoms idxs (JList [j]) = atoms (JInt 0 :: idxs) j
  | atoms idxs (JList (j :: js)) = 
    let
      val atoms1 = atoms (JInt (length js) :: idxs) j
      val atoms2 = atoms idxs (JList js)
    in atoms1 @ atoms2 end
  | atoms idxs j = [(idxs, j)];

fun paths (JSON js) = atoms [] (JObj js);

fun fold f jt s = Basics.fold f (atoms [] jt) s;

fun size jt = 
  let 
    fun add_size (_, jatom) sum = (case jatom of 
      (JString str) => String.size str + sum
    | _ => 1 + sum)
  in fold add_size jt 0 end;

fun damply_via free lim jt =
  let
    fun add_to buff0 str =
      let
        val buff1 = buff0 ^ str
        val buff2 = if String.size buff0 > lim 
          then ((free buff1); "") else buff1
      in buff2 end;
    fun rec_add_to buff i jt  =
      (case jt of
        JList [] => add_to buff "[]" 
        | JList js =>
          let
            val buff = add_to buff "[\n"
            val indent = Symbol.spaces (i + 2)
            fun add_next j (is_first, b) =
              let
                val b = if is_first then b else add_to b ",\n"
                val b = add_to b indent
                val b = rec_add_to b (i + 2) j
              in (false, b) end
            val (_, buff) = Basics.fold add_next js (true, buff)
          in
            add_to buff ("\n" ^ Symbol.spaces i ^ "]")
          end
        | JObj [] => add_to buff "{}"
        | JObj kvs =>
            let
              val buff = add_to buff "{\n"
              val indent = Symbol.spaces (i + 2)
              fun add_next (k, v) (is_first, b) =
                let
                  val b = if is_first then b else add_to b ",\n"
                  val b = add_to b (indent ^ Library.quote k ^ ": ")
                  val b = rec_add_to b (i + 2) v
                in (false, b) end
              val (_, buff) = Basics.fold add_next kvs (true, buff)
            in
              add_to buff ("\n" ^ Symbol.spaces i ^ "}")
            end
      | atom => add_to buff (str_of_atom atom)
    )
  in
    let 
      val final_buff = rec_add_to "" 0 jt 
    in free final_buff end
  end;

fun write_to file_path (JSON jt) =
  let
    val path = Get.os_agnostic_path file_path
    fun write_to stream jt = damply_via (File_Stream.output stream) 4096 jt;
  in
    File_Stream.open_output (fn stream => write_to stream (JObj jt)) path
  end;

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
