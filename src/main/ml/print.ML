(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Utilities for printing Isabelle data to various channels and outputs.
*)

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
  val pretty_json_attrib: bool -> int -> string -> Pretty.T -> Pretty.T
end;


structure Print: PRINT =
struct

datatype mode = FILE of string | OUT;

datatype format = XML of int | YX | AST;

exception FORMAT of string

(* Pretty.T -> string *)
val yx_string_of = Pretty.string_of

(* Pretty.T -> string *)
val string_of_pretty = Get.string_of_yxml o yx_string_of

(* term -> string *)
val term_yx_string = Syntax.string_of_term

(* Proof.context -> format -> term -> string *)
fun term_string ctxt format t =
  let 
    val result = (case format of
        XML depth => Pretty.string_of (XML.pretty depth (YXML.parse (term_yx_string ctxt t)))
      | YX => (Get.string_of_yxml (term_yx_string ctxt t))
      | AST => (ML_Syntax.print_term t)
    )
  in result
  end;

(* mode -> string -> unit *)
(* TODO: add {force=forced} parameter? *)
fun write_to mode str =
  let 
    val _ = (case mode of
        OUT => writeln str
      | FILE name => 
        let 
          val file = TextIO.openAppend(name) 
          val err = Exn.result TextIO.output (file, str ^ "\n")
          val _ = TextIO.closeOut(file)
        in Exn.release err end)
  in () end;

fun out str = write_to OUT str;

fun pretty_out prts = out (string_of_pretty prts);


(** PRETTY PRINTING **)

(* pretty_json_elem: string -> Pretty.T -> Pretty.T *)
fun pretty_json_elem name prt_attribs = 
  Pretty.block [Pretty.quote (Pretty.str name), (Pretty.str ":"), Pretty.brk 1, prt_attribs];

(* pretty_json_attribs: bool -> string -> string -> int -> Pretty.T list -> Pretty.T *)
fun pretty_json_attribs debug encl1 encl2 lvl prts = 
let
  val jblock = if debug
    then let
      fun prt_sep s lvl = [Pretty.str s, Pretty.fbrk, Pretty.brk (lvl+2)];
      val left_prt = [Pretty.str encl1, Pretty.fbrk, Pretty.brk (lvl+2)];
      val right_prt = [Pretty.fbrk, Pretty.brk lvl, Pretty.str encl2];
      val middle_prt = Ops.intersp (prt_sep "," lvl) prts;
      in Pretty.block (left_prt @ middle_prt @ right_prt) end
    else Pretty.block (Pretty.str encl1 :: 
      (Ops.intersp [Pretty.str ",", Pretty.brk 1] prts) @ [Pretty.str encl2])
in jblock end;

(* pretty_json_attrib: bool -> int -> string -> Pretty.T -> Pretty.T *)
fun pretty_json_attrib debug lvl name prt =
  prt
  |> pretty_json_elem name
  |> Library.single
  |> pretty_json_attribs debug "{" "}" lvl;

end;
