
signature PRINT =
sig
  datatype mode = FILE of string | OUT;
  datatype format = XML of int | YX | AST;
  val string_of: Pretty.T -> string
  val term_string: Proof.context -> format -> term -> string
  val write_to: mode -> string -> unit
  val jpair: string -> Pretty.T -> Pretty.T
  val jattribs: bool -> string -> string -> int -> Pretty.T list -> Pretty.T
  val pretty_state_no_facts: bool -> Toplevel.state -> Pretty.T
  val pretty_state: bool -> Proof.context * ((string * int) * thm) list -> Toplevel.state -> Pretty.T
  val proof_state: mode -> bool -> Toplevel.state -> unit
end;


structure Print: PRINT =
struct

datatype mode = FILE of string | OUT;

datatype format = XML of int | YX | AST;

exception FORMAT of string

(* Pretty.T -> string *)
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
      | YX => (YXML.content_of (term_yx_string ctxt t))
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

(** PRETTY PRINTING **)

(* string -> Pretty.T -> Pretty.T *)
fun jpair name prt_attribs = 
  Pretty.block [Pretty.quote (Pretty.str name), (Pretty.str ":"), Pretty.brk 1, prt_attribs];

(* bool -> string -> string -> int -> Pretty.T list -> Pretty.T *)
fun jattribs debug encl1 encl2 lvl prts = 
let
  val jblock = if debug
    then let
      fun prt_sep s lvl = [Pretty.str s, Pretty.fbrk, Pretty.brk (lvl+2)];
      val left_prt = [Pretty.str encl1, Pretty.fbrk, Pretty.brk (lvl+2)];
      val right_prt = [Pretty.fbrk, Pretty.brk lvl, Pretty.str encl2];
      val middle_prt = Data_Ops.intersp (prt_sep "," lvl) prts;
      in Pretty.block (left_prt @ middle_prt @ right_prt) end
    else Pretty.block (Pretty.str encl1 :: 
      (Data_Ops.intersp [Pretty.str ",", Pretty.brk 1] prts) @ [Pretty.str encl2])
in jblock end;


(** GOAL STATE **)


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

(* Proof.context -> (Proof.context -> 'key -> Pretty.T) -> (Proof.context -> 'val -> Pretty.T) 
  -> ('key * 'val list) list -> Pretty.T list *)
fun pretty_alist ctxt pretty_key pretty_val str alist = 
  let
    fun pretty_right k vs = Pretty.block (
      Pretty.str "\"" :: Pretty.commas (map (pretty_val ctxt) vs) 
      @ [Pretty.brk 1, Pretty.str "::", Pretty.brk 1, (pretty_key ctxt) k, Pretty.str "\""]
    );
    fun pretty_entry (n, (k, vs)) = jpair (str^(Value.print_int n)) (pretty_right k vs)
  in map pretty_entry (Data_Ops.enumerate alist) end;

(* Proof.context -> term list -> Pretty.T *)
fun pretty_consts ctxt consts = pretty_alist ctxt
  Syntax.pretty_typ Syntax.pretty_term "Type" (alist_by const_info consts);

(* Proof.context -> term list -> Pretty.T *)
fun pretty_vars ctxt vars = pretty_alist ctxt
  Syntax.pretty_typ Syntax.pretty_term "Type" (alist_by vars_info vars);

(* Proof.context -> typ list -> Pretty.T list *)
fun pretty_types ctxt typs = pretty_alist ctxt
  Syntax.pretty_sort Syntax.pretty_typ "Sort" (alist_by types_info typs);

(* Proof.context -> (string * int) * thm -> Pretty.T list *)
(* TODO: rename *)
fun pretty_named_thm ctxt ((name, i),thm) = [
  jpair "name" (
    Pretty.quote (Pretty.block [
      Pretty.marks_str (#1 (Proof_Context.markup_extern_fact ctxt name), name),
      Pretty.str (Thm_Name.print_suffix (name, i))])
  ),
  jpair "term" (Pretty.quote (Thm.pretty_thm ctxt thm))
  ];

(* Proof.context -> int -> ((string * int) * thm) list -> Pretty.T *)
(* TODO: rename *)
fun pretty_named_thms ctxt lvl named_thms =
  let
    val prts = map (fn named_thm => 
      jpair "thm"
        (jattribs false "{" "}" (~2) (pretty_named_thm ctxt named_thm))) named_thms
  in jpair "thms" (jattribs true "[" "]" lvl prts) end;

in

(* bool -> Toplevel.state -> Pretty.T *)
fun pretty_state_no_facts debug state = 
  let
    val state_data = Get.goal_data state;
    val ctxt = Proof.context_of (Toplevel.proof_of state)
      |> Config.put show_sorts false
      |> Config.put show_types false 
    val Ts = jpair "type variables" (jattribs debug "[" "]" 4 (pretty_types ctxt (#types state_data)));
    val cs = jpair "constants" (jattribs debug "[" "]" 4 (pretty_consts ctxt (#consts state_data)));
    val vs = jpair "variables" (jattribs debug "[" "]" 4 (pretty_vars ctxt (#vars state_data)));
    val goal = jpair "term" (Pretty.quote (Syntax.pretty_term ctxt (#goal state_data)));
  in jattribs debug "{" "}" 0 [jpair "state" (jattribs debug "{" "}" 2 [goal, vs, cs, Ts])] end;

(* bool -> Toplevel.state -> Proof.context * ((string * int) * thm) list -> Pretty.T *)
fun pretty_state debug (ctxt', named_thms) state = 
  let
    val state_data = Get.goal_data state;
    val ctxt = Proof.context_of (Toplevel.proof_of state)
      |> Config.put show_sorts false
      |> Config.put show_types false 
    val Ts = jpair "type variables" (jattribs debug "[" "]" 4 (pretty_types ctxt (#types state_data)));
    val cs = jpair "constants" (jattribs debug "[" "]" 4 (pretty_consts ctxt (#consts state_data)));
    val vs = jpair "variables" (jattribs debug "[" "]" 4 (pretty_vars ctxt (#vars state_data)));
    val goal = jpair "term" (Pretty.quote (Syntax.pretty_term ctxt (#goal state_data)));
    val lems = pretty_named_thms ctxt' 4 named_thms;
  in jattribs debug "{" "}" 0 [jpair "state" (jattribs debug "{" "}" 2 [goal, vs, cs, Ts, lems])] end;

(* mode -> bool -> Toplevel.state -> unit *)
fun proof_state mode debug state = write_to mode (string_of (pretty_state_no_facts debug state));

end;


end;
