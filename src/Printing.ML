
signature PRINTING =
sig
  datatype mode = FILE of string | OUT;
  datatype format = YXML of int | TRM | AST | GOAL;
  val term_string: Proof.context -> format -> term -> string
  val thm_string: Proof.context -> format -> thm -> string
  val print_str: mode -> string -> unit
end;


structure Printing: PRINTING =
struct

datatype mode = FILE of string | OUT;

datatype format = YXML of int | TRM | AST | GOAL;

exception FORMAT of string

val yx_string = Syntax.string_of_term

(* Proof.context -> format -> trm -> string *)
fun term_string ctxt format t =
  let 
    val result = (case format of
        YXML depth => Pretty.string_of (XML.pretty depth (YXML.parse (yx_string ctxt t)))
      | TRM => (YXML.content_of (yx_string ctxt t))
      | AST => (ML_Syntax.print_term t)
      | GOAL => raise FORMAT("Wrong format GOAL for type term.")
    )
  in result
  end;

(* Proof.context -> format -> thm -> string *)
fun thm_string ctxt format thm =
  let 
    val main_term = Thm.prop_of thm;
    val result = (case format of
        GOAL => YXML.content_of (Pretty.string_of (Goal_Display.pretty_goal ctxt thm))
      | _ => term_string ctxt format main_term
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