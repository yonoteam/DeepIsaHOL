
signature DATA_OPS =
sig
  val insert_if: ('a -> bool) -> 'a -> 'a list -> 'a list
  val intersp: 'a list -> 'a list -> 'a list
  val enumerate: 'a list -> (int * 'a) list
  val mk_default_tokens: Proof.context -> string -> Token.T list
  val split_by: string -> string -> string list
  val uniques: ''a list -> ''a list
  val read_file: string -> string
end;


structure Data_Ops: DATA_OPS =
struct

(* GENERIC DATA OPS *)

(* ('a -> bool) -> 'a -> 'a list -> 'a list *)
fun insert_if P x xs = if P x then x :: xs else xs;

(* 'a list -> 'a list -> 'a list *)
fun intersp seps (x :: (xs as  _ :: _)) = (x :: seps) @ (intersp seps xs)
  | intersp _ xs = xs;

(* 'a list -> (int * 'a) list *)
fun enumerate xs =
  let
    fun iter_enum _ [] = []
      | iter_enum n (y::ys) = (n, y) :: (iter_enum (n+1) ys)
  in iter_enum 0 xs end;

(* string -> string -> string list *)
fun split_by split_str target_str = space_explode split_str target_str;

(* 'a list -> 'a list *)
fun uniques xs = Library.distinct (op =) xs;

(* string -> string *)
fun read_file fname = 
  let 
    val IO_instream = TextIO.openIn fname;
    val content = TextIO.inputAll IO_instream;
    val _ = TextIO.closeIn IO_instream;
  in content end;

(* context -> string -> Token.T list *)
fun mk_default_tokens ctxt = Token.explode (Thy_Header.get_keywords' ctxt) Position.none;


end;