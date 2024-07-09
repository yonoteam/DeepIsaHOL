
signature PRINT =
sig
  datatype mode = FILE of string | OUT;
  datatype format = XML of int | XY | AST;
  val string_of: Pretty.T -> string
  val term_string: Proof.context -> format -> term -> string
  val write_to: mode -> string -> unit
  val proof_state: bool -> Proof.context -> Toplevel.state -> string
end;


structure Print: PRINT =
struct

datatype mode = FILE of string | OUT;

datatype format = XML of int | XY | AST;

exception FORMAT of string

val yx_string_of = Pretty.string_of

(* Pretty.T -> string *)
val string_of = YXML.content_of o yx_string_of

(* term -> string *)
val term_yx_string = Syntax.string_of_term

(* Proof.context -> format -> term -> string *)
fun term_string ctxt format t =
  let 
    val result = (case format of
        XML depth => Pretty.string_of (XML.pretty depth (YXML.parse (term_yx_string ctxt t)))
      | XY => (YXML.content_of (term_yx_string ctxt t))
      | AST => (ML_Syntax.print_term t)
    )
  in result
  end;

(* mode -> string -> unit *)
fun write_to mode str =
  let 
    val _ = (case mode of
        OUT => writeln str
      | FILE name => let 
          val file = TextIO.openAppend(name) 
        in TextIO.output(file, str ^ "\n"); TextIO.closeOut(file)
        end)
  in () 
  end;

(* GOAL STATE *)

local

fun not_member_cons_if P x xs = Data_Ops.insert_if (P andf (Pred.neg (member (op =) xs))) x xs;

(* ('a -> ('val, 'key)) -> 'a list -> ('key * 'val list) list *)
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

(* Proof.context -> {show_sorts: bool, show_types: bool} 
  -> (Proof.context -> 'key -> Pretty.T) -> (Proof.context -> 'val -> Pretty.T) 
  -> ('key * 'val list) list -> Pretty.T list *)
(* TODO: Do we need the option to change show_types and show_sorts *)
fun pretty_alist ctxt config pretty_key pretty_val alist = 
  let
    val {show_types = showT, show_sorts = showS} = config;
    val ctxt' = ctxt
      |> Config.put show_sorts showS
      |> Config.put show_types showT
    fun pretty_entry (k, vs) = Pretty.block 
      (Pretty.commas (map (pretty_val ctxt') vs) 
      @ [Pretty.str " ::", Pretty.brk 1, (pretty_key ctxt') k]);
  in map pretty_entry alist end;

(* term -> term * typ *)
fun const_info t = (case t of 
  Const (_, T) => (t, T) 
  | _ => raise TERM("not a constant", [t]));

(* term -> term * typ *)
fun vars_info t = (case t of 
  Var (_, T) => (t,T)  
  | Free (_, T) => (t, T) 
  | _ => raise TERM("not a var", [t]));

(* typ -> typ * sort *)
fun types_info T = (case T of 
  TVar (_, S) => (T, S) 
  | TFree (_, S) => (T, S) 
  | _ => raise TYPE("not a type variable", [T], []));

(* Proof.context -> term list -> Pretty.T *)
fun pretty_consts ctxt consts = pretty_alist ctxt {show_types = true, show_sorts = false} 
  Syntax.pretty_typ Syntax.pretty_term (alist_by const_info consts);

(* Proof.context -> term list -> Pretty.T *)
fun pretty_vars ctxt vars = pretty_alist ctxt {show_types = false, show_sorts = false} 
  Syntax.pretty_typ Syntax.pretty_term (alist_by vars_info vars);

(* Proof.context -> typ list -> Pretty.T list *)
fun pretty_types ctxt typs = pretty_alist ctxt {show_types = false, show_sorts = false} 
  Syntax.pretty_sort Syntax.pretty_typ (alist_by types_info typs)

(* bool -> int -> Pretty.T list -> Pretty.T *)
fun make_jattrib debug lvl prts = 
let
  val make_jblock = if debug
    then let
      fun prt_sep s lvl = [Pretty.str s, Pretty.fbrk, Pretty.brk (lvl+2)];
      val left_prt = [Pretty.str "{", Pretty.fbrk, Pretty.brk (lvl+2)];
      val right_prt = [Pretty.fbrk, Pretty.brk lvl, Pretty.str "}"];
      fun middle_prt lvl = Data_Ops.intersp (prt_sep "," lvl);
      in Pretty.block (left_prt @ middle_prt lvl prts @ right_prt) end
    else Pretty.block (Pretty.str "{" :: 
      (Data_Ops.intersp [Pretty.str ",", Pretty.brk 1] prts) @ [Pretty.str "}"])
in make_jblock end;

in

(* bool -> Proof.context -> Toplevel.state -> string *)
fun proof_state debug ctxt state = 
  let
    val state_data = Data_Ops.get_goal_data state;
    fun jpair name prt_attribs = 
      Pretty.block (Pretty.quote (Pretty.str name) :: (Pretty.str ":") :: [Pretty.brk 1, prt_attribs]);
    fun Ts lvl = jpair "type variables" (make_jattrib debug lvl (pretty_types ctxt (#types state_data)));
    fun cs lvl = jpair "constants" (make_jattrib debug lvl (pretty_consts ctxt (#consts state_data)));
    fun vs lvl = jpair "variables" (make_jattrib debug lvl (pretty_vars ctxt (#vars state_data)));
    val goal = jpair "term" (Syntax.pretty_term ctxt (#goal state_data));
  in string_of (jpair "state" (make_jattrib debug 0 [goal, vs 2, cs 2, Ts 2])) end;

end;


end;