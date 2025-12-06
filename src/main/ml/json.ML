(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Internal representation of JSONs as association lists.
*)

signature JAVA_SCRIPT_OBJ_NOTAT =
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
  val upd: string * jtype -> T -> T
  val del: string -> T -> T
  val get: string -> T -> jtype option
  val get_or_else: string -> (jtype -> 'a) -> 'a -> T -> 'a
  val paths: T -> (jtype list * jtype) list
  val fold: (jtype list * jtype -> 'a -> 'a) -> jtype -> 'a -> 'a
  val buffered_fold: int -> (string -> 'a -> 'a) -> jtype -> 'a -> 'a
  val to_string: T -> string
  val write_to: string -> T -> unit
end;

structure JSON: JAVA_SCRIPT_OBJ_NOTAT =
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

fun upd (key, value) (JSON js) = JSON (AList.update (op =) (key, value) js);

fun del key (JSON js) = JSON (AList.delete (op =) key js);

fun get key (JSON js) = AList.lookup (op =) js key;

fun get_or_else key to_val default (JSON js) =
  (case get key (JSON js) of
    SOME v => to_val v
    | NONE => default);

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

fun buffered_fold lim (flush:string -> 'a -> 'a) jt st =
  let
    fun add_to (buff, st) str =
      let
        val buff' = buff ^ str
        val result = if String.size buff' > lim 
          then ("", flush buff' st)
          else (buff', st)
      in result end;
    fun rec_add_to (buff, st) i jt  =
      (case jt of
        JList [] => add_to (buff, st) "[]" 
        | JList js =>
          let
            val (buff, st) = add_to (buff, st) "[\n"
            val indent = Symbol.spaces (i + 2)
            fun add_next j (is_first, b, s) =
              let
                val (b, s) = if is_first then (b, s) else add_to (b, s) ",\n"
                val (b, s) = add_to (b, s) indent
                val (b, s) = rec_add_to (b, s) (i + 2) j
              in (false, b, s) end
            val (_, buff, st) = Basics.fold add_next js (true, buff, st)
          in
            add_to (buff, st) ("\n" ^ Symbol.spaces i ^ "]")
          end
        | JObj [] => add_to (buff, st) "{}"
        | JObj kvs =>
            let
              val (buff, st) = add_to (buff, st) "{\n"
              val indent = Symbol.spaces (i + 2)
              fun add_next (k, v) (is_first, b, s) =
                let
                  val (b, s) = if is_first then (b, s) else add_to (b, s) ",\n"
                  val (b, s) = add_to (b, s) (indent ^ Library.quote k ^ ": ")
                  val (b, s) = rec_add_to (b, s) (i + 2) v
                in (false, b, s) end
              val (_, buff, st) = Basics.fold add_next kvs (true, buff, st)
            in
              add_to (buff, st) ("\n" ^ Symbol.spaces i ^ "}")
            end
      | atom => add_to (buff, st) (str_of_atom atom)
    )
  in
    let 
      val (final_buff, final_st) = rec_add_to ("", st) 0 jt ;
      val result = if final_buff = "" then final_st else flush final_buff final_st;
    in result end
  end;

fun write_to file_path (JSON jt) =
  let
    val path = Get.os_agnostic_path file_path;
    fun flush_to stream = (fn str => fn () => File_Stream.output stream str);
    fun write_to stream jt = buffered_fold 4096 (flush_to stream) jt ();
  in
    File_Stream.open_output (fn stream => write_to stream (JObj jt)) path
  end;

fun str_of_jtype max_depth jt =
  let
    fun str_list _ _ [] = ""
      | str_list to_str i [x] = Symbol.spaces i ^ (to_str x)
      | str_list to_str i (x :: xs) = 
          Symbol.spaces i ^ (to_str x) ^ ",\n" ^ str_list to_str i xs;
    fun rec_str depth i jt =
      (case jt of 
        JList [] => "[]"
        | JList js => 
          if depth >= max_depth then Library.enclose "[" "]" "..."
          else let
            val mid = str_list (rec_str (depth + 1) (i+2)) (i+2) js
          in Library.enclose "[\n" ("\n" ^ Symbol.spaces i ^ "]") mid end
        | JObj [] => "{}"
        | JObj js => 
          if depth >= max_depth then Library.enclose "{" "}" "..."
          else let
            fun to_str (k,v) = Library.quote k ^ ": " ^ (rec_str (depth + 1) (i+2) v);
            val mid = str_list to_str (i+2) js
          in Library.enclose "{\n" ("\n" ^ Symbol.spaces i ^ "}") mid end
        | atom => str_of_atom atom
      );
  in rec_str 0 0 jt end

fun to_string (JSON js) = str_of_jtype 3 (JObj js);

val _ = ML_system_pp (fn _ => fn _  => ML_Pretty.str o to_string);

end;
