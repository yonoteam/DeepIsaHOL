(* Mantainers: 
    Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz

Part of project DeepIsaHOL. Utilities for printing Isabelle data to various channels and outputs.
*)

signature PRINT =
sig
  val yxml_to_string: string -> string
  val string_of_pretty: Pretty.T -> string
  val string_of_term: Proof.context -> term -> string
  val string_of_type: Proof.context -> typ -> string
  val string_of_sort: Proof.context -> sort -> string
  val ast_of_term: term -> string
  val write_to_file: string -> string -> unit
  val coalesce_method_txt: string list -> string
  val make_parseable: string -> string
  val pretty_json_elem: string -> Pretty.T -> Pretty.T
  val pretty_json_attribs: bool -> string -> string -> int -> Pretty.T list -> Pretty.T
  val pretty_json_attrib: bool -> int -> string -> Pretty.T -> Pretty.T
end;


structure Print: PRINT =
struct

val yxml_to_string = YXML.parse_body #> XML.content_of;

val string_of_pretty = yxml_to_string o Pretty.string_of

fun string_of_term ctxt t = 
  Syntax.string_of_term ctxt t
  |> Protocol_Message.clean_output;

fun string_of_type ctxt T = 
  Syntax.string_of_typ ctxt T
  |> Protocol_Message.clean_output;

fun string_of_sort ctxt T = 
  Syntax.string_of_sort ctxt T
  |> Protocol_Message.clean_output;

val ast_of_term = ML_Syntax.print_term;

fun write_to_file file_name str =
  let
    val file = TextIO.openAppend file_name;
    val err = Exn.result TextIO.output (file, str ^ "\n");
    val _ = TextIO.closeOut(file)
  in Exn.release err end;

fun make_parseable txt =
  let
    val backslash = "\\";
    val new_line = "\n";
    val quote = "\"";
    val str_quote = backslash ^ quote;
    val isa_quote = backslash ^ "<";
  in Ops.replace_all quote (str_quote) txt
    |> Ops.replace_all isa_quote (backslash ^ isa_quote)
    |> Ops.replace_all new_line "\\n"
    |> Library.enclose quote quote
  end;

fun coalesce_method_txt [] = ""
  | coalesce_method_txt [s] = s
  | coalesce_method_txt (s1 :: s2 :: ss) =
      if Library.member (op =) ["(", "["] s1
        orelse Library.member (op =) [")", "]", ":", ";"] s2
      then s1 ^ coalesce_method_txt (s2 :: ss)
      else s1 ^ " " ^ coalesce_method_txt (s2 :: ss);


(** PRETTY PRINTING **)

fun pretty_json_elem name prt_attribs = 
  Pretty.block [Pretty.quote (Pretty.str name), (Pretty.str ":"), Pretty.brk 1, prt_attribs];

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

fun pretty_json_attrib debug lvl name prt =
  prt
  |> pretty_json_elem name
  |> Library.single
  |> pretty_json_attribs debug "{" "}" lvl;

end;
