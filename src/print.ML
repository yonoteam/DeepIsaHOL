
signature PRINT =
sig
  datatype mode = FILE of string | OUT;
  datatype format = XML of int | YX | AST;
  val string_of_pretty: Pretty.T -> string
  val term_string: Proof.context -> format -> term -> string
  val write_to: mode -> string -> unit
  val out: string -> unit
  val pretty_out: Pretty.T -> unit
  val pretty_json_elem: string -> Pretty.T -> Pretty.T
  val pretty_json_attribs: bool -> string -> string -> int -> Pretty.T list -> Pretty.T
  (* val pretty_state_no_facts: bool -> Toplevel.state -> Pretty.T
  val pretty_state: bool -> Proof.context * ((string * int) * thm) list -> Toplevel.state -> Pretty.T
  val proof_state: mode -> bool -> Toplevel.state -> unit *)
end;


structure Print: PRINT =
struct

datatype mode = FILE of string | OUT;

datatype format = XML of int | YX | AST;

exception FORMAT of string

(* Pretty.T -> string *)
val yx_string_of = Pretty.string_of

(* Pretty.T -> string *)
val string_of_pretty = YXML.content_of o yx_string_of

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

fun out str = write_to OUT str;

fun pretty_out prts = out (string_of_pretty prts);

(** PRETTY PRINTING **)

(* string -> Pretty.T -> Pretty.T *)
fun pretty_json_elem name prt_attribs = 
  Pretty.block [Pretty.quote (Pretty.str name), (Pretty.str ":"), Pretty.brk 1, prt_attribs];

(* bool -> string -> string -> int -> Pretty.T list -> Pretty.T *)
fun pretty_json_attribs debug encl1 encl2 lvl prts = 
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



end;
