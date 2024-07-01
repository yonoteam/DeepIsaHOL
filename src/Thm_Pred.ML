
signature THM_PRED =
sig
  val neg: ('a -> bool) -> 'a -> bool
  val has_many: string * thm list -> bool
  val nth_letter_is: int -> string -> string * thm list -> bool
  val starts_with: string -> string * thm list -> bool
  val contains: string -> string * thm list -> bool
  val is_dynamic: Proof.context -> string * thm list -> bool
  val passes_fact_check: Proof.context -> string -> bool
end;


structure Thm_Pred: THM_PRED =
struct

fun neg P x = not (P x)

fun has_many (_:string, thms: thm list) = length thms > 1;

fun starts_with str (name, _: thm list) = String.isPrefix str name;

fun nth_letter_is n str (name, _: thm list) = nth_string name n = str;

fun contains str (name, _: thm list) =  String.isSubstring str name;

fun is_dynamic ctxt (name:string, _: thm list) 
  = (#dynamic (Facts.retrieve (Context.Proof ctxt) (Proof_Context.facts_of ctxt) (name, Position.none)))

fun passes_fact_check ctxt name =
let
  val generic = Context.Proof ctxt
  val facts = #local_f (Data_Ops.get_facts ctxt);
  val checks 
    = (case (Facts.check generic facts (name, Position.none)) 
      of _ => true)
    handle ERROR _ => false
in
  checks
end;

end;