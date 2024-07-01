
signature PRINTING =
sig
  datatype mode = FILE of string | OUT;
  datatype format = YXML of int | TRM | AST | GOAL;
  val to_string: Proof.context -> format -> thm -> string
  val print_str: mode -> string -> unit
end;


structure Printing: PRINTING =
struct

datatype mode = FILE of string | OUT;

datatype format = YXML of int | TRM | AST | GOAL;

val yx_string_of = Syntax.string_of_term

(* Proof.context -> format -> thm -> string *)
fun to_string ctxt format thm =
  let 
    val main_term = Thm.prop_of thm;
    val result = (case format of
        YXML depth => Pretty.string_of (XML.pretty depth (YXML.parse (yx_string_of ctxt main_term)))
      | TRM => (YXML.content_of (yx_string_of ctxt main_term))
      | AST => (ML_Syntax.print_term main_term)
      | GOAL => YXML.content_of (Pretty.string_of (Goal_Display.pretty_goal ctxt thm))
    )
  in result
  end;

(* mode -> string -> unit *)
fun print_str mode str =
  let 
    val _ = (case mode of
        OUT => writeln str
      | FILE name => let 
          val file = TextIO.openAppend(name) 
        in TextIO.output(file, str ^ "\n"); TextIO.closeOut(file)
        end)
  in () 
  end;

end;