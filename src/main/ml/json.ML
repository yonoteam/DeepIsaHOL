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

fun get_or_else key to_val default (JSON js) =
  (case get key (JSON js) of
    SOME v => to_val v
    | NONE => default);

fun make_jtype_list name to_jtyp vs =
  map (fn v => JObj [(name, to_jtyp v)]) vs;

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

fun print buff free lim i jt =
  let
    fun safely_add buff' str =
      let
        val buff'' = buff' ^ str
        val result = if String.size buff' > lim 
          then ((free buff''); "")
          else buff''
      in result end;
    fun rec_print i buff (JList xs) =
        let
          val lead = "\n" ^ Symbol.spaces (i+2);
          fun add_all [] buff' = buff'
            | add_all [j] buff' = 
              if size j > lim then 
                (free (buff' ^ lead); rec_print (i+2) "" j)
              else safely_add (buff' ^ lead) (str_of_jtype (i+2) j)
            | add_all (j::js) buff' =
              if size j > lim then 
                (free (buff' ^ lead); add_all js ((rec_print (i+2) "" j) ^ ", "))
              else add_all js (safely_add (buff' ^ lead) ((str_of_jtype (i+2) j) ^ ", "));
          val result = if null xs then "[]" else 
            let
              val buff' = safely_add buff "[" |> add_all xs;
            in safely_add buff' ("\n" ^ Symbol.spaces i ^ "]") end;
        in result end
      | rec_print i buff (JObj js) =
        let
          fun lead k = "\n" ^ Symbol.spaces (i+2) ^ Library.quote k ^ ": ";
          fun add_all [] buff' = buff'
            | add_all [(k,v)] buff' = 
              if size v > lim then 
                (free (buff' ^ lead k); rec_print (i+2) "" v)
              else safely_add (buff' ^ lead k) (str_of_jtype (i+2) v)
            | add_all ((k,v)::js) buff' =
              if size v > lim then 
                (free (buff' ^ lead k); add_all js ((rec_print (i+2) "" v) ^ ", "))
              else add_all js (safely_add (buff' ^ lead k) ((str_of_jtype (i+2) v) ^ ", "));
          val result = if null js then "{}" else 
            let
              val buff' = safely_add buff "{" |> add_all js;
            in safely_add buff' ("\n" ^ Symbol.spaces i ^ "}") end;
        in result end
      | rec_print i buff (JString str) =
        let
          val pre_result = str_of_jtype i (JString str)
          val result = if String.size pre_result > lim
            then ((free buff); (free pre_result); "")
            else safely_add buff (str_of_jtype i (JString str))
        in result end
      | rec_print i buff jt = safely_add buff (str_of_jtype i jt);
  in free (rec_print i buff jt) end;

fun write_to file_path (JSON jt)
  = print "" (File.append (Get.os_agnostic_path file_path)) 10000 0 (JObj jt)

end;
